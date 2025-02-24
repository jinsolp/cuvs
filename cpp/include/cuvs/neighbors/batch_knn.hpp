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

#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/host_mdarray.hpp>

#include <variant>

namespace cuvs::neighbors::batch_knn {

enum knn_build_algo { NN_DESCENT, IVF_PQ };

template <typename IdxT, typename DistT = float>
struct index : cuvs::neighbors::index {
 public:
  index(raft::resources const& res,
        int64_t n_rows,
        int64_t k,
        bool return_distances               = false,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
        knn_build_algo build_algo           = NN_DESCENT,
        size_t n_clusters                   = 4)
    : cuvs::neighbors::index(),
      res{res},
      k{k},
      metric{metric},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(n_rows, k)},
      graph_view_{graph_.view()},
      return_distances{return_distances},
      build_algo{build_algo},
      n_clusters{n_clusters}
  {
    if (return_distances) {
      distances_      = raft::make_device_matrix<DistT, int64_t>(res, n_rows, k);
      distances_view_ = distances_.value().view();
    }
  }

  index(raft::resources const& res,
        raft::host_matrix_view<IdxT, int64_t, raft::row_major> graph_view,
        std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view =
          std::nullopt,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded,
        knn_build_algo build_algo           = NN_DESCENT,
        size_t n_clusters                   = 4)
    : cuvs::neighbors::index(),
      res{res},
      k{graph_view.extent(1)},
      metric{metric},
      graph_{raft::make_host_matrix<IdxT, int64_t, raft::row_major>(0, 0)},
      graph_view_{graph_view},
      distances_view_{distances_view},
      return_distances{distances_view.has_value()},
      build_algo{build_algo},
      n_clusters{n_clusters}
  {
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() noexcept
    -> raft::host_matrix_view<IdxT, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  /** neighborhood graph distances [size, graph-degree] */
  [[nodiscard]] inline auto distances() noexcept
    -> std::optional<device_matrix_view<float, int64_t, row_major>>
  {
    return distances_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  raft::resources const& res;
  int64_t k;
  cuvs::distance::DistanceType metric;
  bool return_distances;
  knn_build_algo build_algo;
  size_t n_clusters;

 private:
  // raft::resources const& res_;
  // int64_t k;
  // cuvs::distance::DistanceType metric_;
  raft::host_matrix<IdxT, int64_t, raft::row_major> graph_;  // graph to return for non-int IdxT
  std::optional<raft::device_matrix<float, int64_t, row_major>> distances_;
  raft::host_matrix_view<IdxT, int64_t, raft::row_major>
    graph_view_;  // view of graph for user provided matrix
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances_view_;
  // bool return_distances_;
  // knn_build_algo build_algo;
  // size_t n_clusters;
};

/**
 * @brief ANN parameters used by CAGRA to build knn graph
 *
 */
namespace graph_build_params {

/** Specialized parameters utilizing IVF-PQ to build knn graph */
struct ivf_pq_params {
  cuvs::neighbors::ivf_pq::index_params build_params;
  cuvs::neighbors::ivf_pq::search_params search_params;
  float refinement_rate;

  ivf_pq_params() = default;
  /**
   * Set default parameters based on shape of the input dataset.
   * Usage example:
   * @code{.cpp}
   *   using namespace cuvs::neighbors;
   *   raft::resources res;
   *   // create index_params for a [N. D] dataset
   *   auto dataset = raft::make_device_matrix<float, int64_t>(res, N, D);
   *   auto pq_params =
   *     cagra::graph_build_params::ivf_pq_params(dataset.extents());
   *   // modify/update index_params as needed
   *   pq_params.kmeans_trainset_fraction = 0.1;
   * @endcode
   */
  ivf_pq_params(raft::matrix_extent<int64_t> dataset_extents,
                cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);
};

using nn_descent_params = cuvs::neighbors::nn_descent::index_params;
}  // namespace graph_build_params

struct index_params : cuvs::neighbors::index_params {
  /** Parameters for graph building.
   *
   * Set ivf_pq_params, nn_descent_params, or iterative_search_params to select the graph build
   * algorithm and control their parameters. The default (std::monostate) is to use a heuristic
   *  to decide the algorithm and its parameters.
   *
   * @code{.cpp}
   * cagra::index_params params;
   * // 1. Choose IVF-PQ algorithm
   * params.graph_build_params = cagra::graph_build_params::ivf_pq_params(dataset.extent,
   * params.metric);
   *
   * // 2. Choose NN Descent algorithm for kNN graph construction
   * params.graph_build_params =
   * cagra::graph_build_params::nn_descent_params(params.intermediate_graph_degree);
   *
   * // 3. Choose iterative graph building using CAGRA's search() and optimize()  [Experimental]
   * params.graph_build_params =
   * cagra::graph_build_params::iterative_search_params();
   * @endcode
   */
  std::variant<graph_build_params::nn_descent_params, graph_build_params::ivf_pq_params>
    graph_build_params;

  bool attach_dataset_on_build = true;
};

auto build(const raft::resources& handle,
           raft::host_matrix_view<const float, int64_t, row_major> index_dataset,
           int64_t k,
           const index_params& params) -> index<int64_t, float>;

}  // namespace cuvs::neighbors::batch_knn
