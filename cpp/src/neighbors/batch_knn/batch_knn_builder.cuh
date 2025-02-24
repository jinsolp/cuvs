/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
//  #include "batch_knn.cuh"
//  #include <cuvs/neighbors/batch_knn.hpp>
#include "../detail/nn_descent.cuh"
#include "cuvs/neighbors/ivf_pq.hpp"
#include "cuvs/neighbors/nn_descent.hpp"
#include <raft/matrix/sample_rows.cuh>
//  #include "batch_knn_common.cuh"
#include "cuvs/neighbors/batch_knn.hpp"

namespace cuvs::neighbors::batch_knn::detail {
using namespace cuvs::neighbors;
using align32 = raft::Pow2<32>;

template <typename T, typename IdxT = int64_t>
struct batch_knn_builder {
  batch_knn_builder() {}

  virtual void prepare_build(raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    // preparing build (index of making gnnd etc)
  }

  virtual void build_knn()
  {
    // build for nnd, search and refinement for ivfpq
  }
};

template <typename T, typename IdxT = int64_t>
struct batch_knn_builder_ivfpq : public batch_knn_builder<T, IdxT> {
  batch_knn_builder_ivfpq(raft::resources const& res,
                          size_t n_clusters,
                          batch_knn::graph_build_params::ivf_pq_params& params)
    : batch_knn_builder<T, IdxT>(),
      res{res},
      index_params{params.build_params},
      search_params{params.search_params}
  {
    // index_params    = params.build_params;
    // search_params   = params.search_params;
    // index = ivf_pq::index<IdxT>(res);
    index_params.add_data_on_build = false;
    refinement_rate                = 2.0;
    n_clusters                     = n_clusters;
  }

  void prepare_build(raft::host_matrix_view<const T, int64_t, row_major> dataset) override
  {
    // build ivf-pq index on a random subset for efficient GPU memory usage
    size_t num_rows = static_cast<size_t>(dataset.extent(0));
    size_t num_cols = static_cast<size_t>(dataset.extent(1));
    size_t num_subsamples =
      std::min(static_cast<size_t>(num_rows / n_clusters), static_cast<size_t>(num_rows * 0.1));

    auto d_dataset_subsample =
      raft::make_device_matrix<T, int64_t, row_major>(res, num_subsamples, num_cols);
    raft::matrix::sample_rows<T, int64_t>(
      res, raft::random::RngState{0}, dataset, d_dataset_subsample.view());

    // index = ivf_pq::index<IdxT>(res, index_params, dataset.extent(1));

    // auto ress = cuvs::neighbors::ivf_pq::build(res, index_params,
    // raft::make_const_mdspan(d_dataset_subsample.view()));
    index.emplace(cuvs::neighbors::ivf_pq::build(
      res, index_params, raft::make_const_mdspan(d_dataset_subsample.view())));
  }

  void build_knn() override
  {
    // build for nnd, search and refinement for ivfpq
    std::cout << "attempting to build knn with overrode function in ivfpq builder\n";
  }

  // TODO: index doesn't need init?
  ivf_pq::index_params& index_params;
  ivf_pq::search_params& search_params;
  float refinement_rate;
  std::optional<ivf_pq::index<IdxT>> index;
  raft::resources const& res;
  size_t n_clusters;
};

// void build_index(
//   raft::resources const& res,
//   const ivf_pq::index_params& index_params,
//   ivf_pq::index<IdxT>* index,
//   raft::host_matrix_view<const T, int64_t, row_major> dataset,
//     size_t n_clusters)
// {
//     // builds the index on a subsample of the dataset for efficient GPU memory usage
//   size_t num_rows   = static_cast<size_t>(dataset.extent(0));
//   size_t num_cols   = static_cast<size_t>(dataset.extent(1));
//   size_t num_subsamples =
//     std::min(static_cast<size_t>(num_rows / n_clusters), static_cast<size_t>(num_rows * 0.1));

//   auto d_dataset_subsample =
//     raft::make_device_matrix<T, int64_t>(res, num_subsamples, num_cols);
//   raft::matrix::sample_rows<T, int64_t>(
//     res, raft::random::RngState{0}, dataset, d_dataset_subsample.view());

//     cuvs::neighbors::ivf_pq::build(res, index_params, d_dataset_subsample, index);
// }

template <typename T, typename IdxT = int64_t>
struct batch_knn_builder_nn_descent : public batch_knn_builder<T, IdxT> {
  batch_knn_builder_nn_descent(raft::resources const& res,
                               size_t n_clusters,
                               batch_knn::graph_build_params::nn_descent_params& index_params,
                               size_t min_cluster_size,
                               size_t max_cluster_size)
    : batch_knn_builder<T, IdxT>(), res{res}
  {
    n_clusters       = n_clusters;
    index_params     = index_params;
    min_cluster_size = min_cluster_size;
    max_cluster_size = max_cluster_size;
  }

  void prepare_build(raft::host_matrix_view<const T, int64_t, row_major> dataset) override
  {
    size_t intermediate_degree = index_params.intermediate_graph_degree;
    size_t graph_degree        = index_params.graph_degree;

    if (intermediate_degree >= min_cluster_size) { intermediate_degree = min_cluster_size - 1; }

    if (intermediate_degree < graph_degree) {
      RAFT_LOG_WARN(
        "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
        "graph_degree.",
        graph_degree,
        intermediate_degree);
      graph_degree = intermediate_degree;
    }

    size_t extended_graph_degree =
      align32::roundUp(static_cast<size_t>(graph_degree * (graph_degree <= 32 ? 1.0 : 1.3)));
    size_t extended_intermediate_degree = align32::roundUp(
      static_cast<size_t>(intermediate_degree * (intermediate_degree <= 32 ? 1.0 : 1.3)));

    build_config.max_dataset_size      = max_cluster_size;
    build_config.dataset_dim           = dataset.extent(1);
    build_config.node_degree           = extended_graph_degree;
    build_config.internal_node_degree  = extended_intermediate_degree;
    build_config.max_iterations        = index_params.max_iterations;
    build_config.termination_threshold = index_params.termination_threshold;
    build_config.output_graph_degree   = graph_degree;

    if (!nnd_builder.has_value()) {
      // Initialize nnd_builder in the first call to prepare_build
      nnd_builder.emplace(res, build_config);
    }
    // nnd_builder = nn_descent::detail::GNND<const T, int>(res, build_config);
  }

  void build_knn() override
  {
    // build for nnd, search and refinement for ivfpq
    std::cout << "attempting to build knn with overrode function in nnd builder\n";
  }

  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size;
  nn_descent::index_params index_params;
  nn_descent::detail::BuildConfig build_config;

  std::optional<nn_descent::detail::GNND<const T, int>> nnd_builder;
  // nn_descent::detail::GNND<const T, int> nnd;
};

}  // namespace cuvs::neighbors::batch_knn::detail
