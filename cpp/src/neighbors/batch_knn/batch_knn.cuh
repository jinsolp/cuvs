/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "../detail/ann_utils.cuh"
#include "cuvs/neighbors/common.hpp"
#include "cuvs/neighbors/ivf_pq.hpp"
#include <cstddef>
#include <cuvs/neighbors/batch_knn.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/resource/nccl_clique.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <raft/matrix/sample_rows.cuh>

#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/util/cudart_utils.hpp>
#include <variant>

#include "batch_knn_builder.cuh"
#include <cuvs/neighbors/batch_knn.hpp>

// #include "batch_knn_builder.cuh"

namespace cuvs::neighbors::batch_knn::detail {
using namespace cuvs::neighbors;

// using namespace raft;

// template <typename IdxT>
// void write_to_graph(raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
//                     raft::host_matrix_view<int64_t, int64_t, raft::row_major>
//                     neighbors_host_view, size_t& num_self_included, size_t batch_size, size_t
//                     batch_offset)
// {
//   uint32_t node_degree = knn_graph.extent(1);
//   size_t top_k         = neighbors_host_view.extent(1);
//   // omit itself & write out
//   for (std::size_t i = 0; i < batch_size; i++) {
//     size_t vec_idx = i + batch_offset;
//     for (std::size_t j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
//       const auto v = neighbors_host_view(i, j);
//       if (static_cast<size_t>(v) == vec_idx) {
//         num_self_included++;
//         continue;
//       }
//       knn_graph(vec_idx, num_added) = v;
//       num_added++;
//     }
//   }
// }

// template <typename DataT, typename IdxT, typename accessor>
// void refine_host_and_write_graph(
//   raft::resources const& res,
//   raft::host_matrix<DataT, int64_t>& queries_host,
//   raft::host_matrix<int64_t, int64_t>& neighbors_host,
//   raft::host_matrix<int64_t, int64_t>& refined_neighbors_host,
//   raft::host_matrix<float, int64_t>& refined_distances_host,
//   raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
//   raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
//   cuvs::distance::DistanceType metric,
//   size_t& num_self_included,
//   size_t batch_size,
//   size_t batch_offset,
//   int top_k,
//   int gpu_top_k)
// {
//   bool do_refine = top_k != gpu_top_k;

//   auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
//     do_refine ? refined_neighbors_host.data_handle() : neighbors_host.data_handle(),
//     batch_size,
//     top_k);

//   if (do_refine) {
//     // needed for compilation as this routine will also be run for device data with !do_refine
//     if constexpr (raft::is_host_mdspan_v<decltype(dataset)>) {
//       auto queries_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
//         queries_host.data_handle(), batch_size, dataset.extent(1));
//       auto neighbors_host_view = raft::make_host_matrix_view<const int64_t, int64_t>(
//         neighbors_host.data_handle(), batch_size, neighbors_host.extent(1));
//       auto refined_distances_host_view = raft::make_host_matrix_view<float, int64_t>(
//         refined_distances_host.data_handle(), batch_size, top_k);
//       cuvs::neighbors::refine(res,
//                               dataset,
//                               queries_host_view,
//                               neighbors_host_view,
//                               refined_neighbors_host_view,
//                               refined_distances_host_view,
//                               metric);
//     }
//   }

//   write_to_graph(
//     knn_graph, refined_neighbors_host_view, num_self_included, batch_size, batch_offset);
// }

// // maybe change all the distances to float?
// template <typename T, typename IdxT>
// void ivf_pq_single_gpu(
//   raft::resources const& res,
//   raft::host_matrix_view<const T, int64_t, row_major> dataset,
//   batch_knn::index<IdxT, T> & index,
//   const ivf_pq::index_params* index_params,
// const ivf_pq::search_params* search_params = std::nullptr_t)
// {
// //   RAFT_EXPECTS(index.metric == cuvs::distance::DistanceType::L2Expanded ||
// //     index.metric == cuvs::distance::DistanceType::InnerProduct,
// //                "Currently only L2Expanded or InnerProduct metric are supported");

//   int64_t k = index.k;

//   // TODO: this part also needs to be done on a subset of the data
//   auto ivf_pq_index = ivf_pq::build(res, index_params, dataset);
//   float refinement_rate = 2.0;  // TODO change refinement rate?

//   const auto top_k       = k + 1;
//   int64_t gpu_top_k     = k * refinement_rate;
//   gpu_top_k              = std::min(std::max(gpu_top_k, top_k), dataset.extent(0));
//   const auto num_queries = dataset.extent(0);

//   // Use the same maximum batch size as the ivf_pq::search to avoid allocating more than needed.
//   constexpr uint32_t kMaxQueries = 4096;

//   auto d_distances = raft::make_device_matrix<T>(res, kMaxQueries, gpu_top_k);
//   auto d_neighbors = raft::make_device_matrix<IdxT>(res, kMaxQueries, gpu_top_k);
//   auto d_refined_distances = raft::make_device_matrix<T>(res, kMaxQueries, top_k);
//   auto d_refined_neighbors = raft::make_device_matrix<IdxT>(res, kMaxQueries, top_k);

//   auto h_queries = raft::make_host_matrix<T>(kMaxQueries, dataset.extent(1));
//   auto h_neighbors = raft::make_host_matrix<T>(kMaxQueries, gpu_top_k);
//   auto h_refined_distances = raft::make_host_matrix<T>(kMaxQueries, top_k);
//   auto h_refined_neighbors = raft::make_host_matrix<IdxT>(kMaxQueries, top_k);

//   std::size_t num_self_included = 0;

//   spatial::knn::detail::utils::batch_load_iterator<T> vec_batches(
//     dataset.data_handle(),
//     dataset.extent(0),
//     dataset.extent(1),
//     static_cast<int64_t>(kMaxQueries),
//     raft::resource::get_cuda_stream(res));

//   size_t next_report_offset = 0;
//   size_t d_report_offset    = dataset.extent(0) / 100;  // Report progress in 1% steps.

//   bool async_host_processing   = raft::is_host_mdspan_v<decltype(dataset)> || top_k == gpu_top_k;
//   size_t previous_batch_size   = 0;
//   size_t previous_batch_offset = 0;

//   for (const auto& batch : vec_batches) {
//     // Map int64_t to uint32_t because ivf_pq requires the latter.
//     auto d_queries_view = raft::make_device_matrix_view<const T, uint32_t>(
//       batch.data(), batch.size(), batch.row_width());
//     auto d_neighbors_view = raft::make_device_matrix_view<IdxT, uint32_t>(
//       d_neighbors.data_handle(), batch.size(), gpu_top_k);
//     auto d_distances_view = raft::make_device_matrix_view<T, uint32_t>(
//       d_distances.data_handle(), batch.size(), gpu_top_k);

//       // search is done on a batch??
//     cuvs::neighbors::ivf_pq::search(
//       res, search_params, ivf_pq_index, d_queries_view, d_neighbors_view, d_distances_view);

//      // process previous batch async on host
//       // NOTE: the async path also covers disabled refinement (top_k == gpu_top_k)
//       if (previous_batch_size > 0) {
//         refine_host_and_write_graph(res,
//                                     h_queries,
//                                     h_neighbors,
//                                     h_refined_neighbors,
//                                     h_refined_distances,
//                                     dataset,
//                                     index.graph(),
//                                     index.metric,
//                                     num_self_included,
//                                     previous_batch_size,
//                                     previous_batch_offset,
//                                     top_k,
//                                     gpu_top_k);
//       }

//       // copy next batch to host
//       raft::copy(h_neighbors.data_handle(),
//                  d_neighbors.data_handle(),
//                  d_neighbors_view.size(),
//                  raft::resource::get_cuda_stream(res));
//       if (top_k != gpu_top_k) {
//         // can be skipped for disabled refinement
//         raft::copy(h_queries.data_handle(),
//                    batch.data(),
//                    d_queries_view.size(),
//                    raft::resource::get_cuda_stream(res));
//       }

//       previous_batch_size   = batch.size();
//       previous_batch_offset = batch.offset();

//       // we need to ensure the copy operations are done prior using the host data
//       raft::resource::sync_stream(res);

//       // process last batch
//       if (previous_batch_offset + previous_batch_size == (size_t)num_queries) {
//         refine_host_and_write_graph(res,
//                                     h_queries,
//                                     h_neighbors,
//                                     h_refined_neighbors,
//                                     h_refined_distances,
//                                     dataset,
//                                     index.graph(),
//                                     index.metric,
//                                     num_self_included,
//                                     previous_batch_size,
//                                     previous_batch_offset,
//                                     top_k,
//                                     gpu_top_k);
//       }

//     //   // do refinement
//     //   auto neighbor_candidates_view = raft::make_device_matrix_view<const IdxT, uint64_t>(
//     //     neighbors.data_handle(), batch.size(), gpu_top_k);
//     //   auto refined_neighbors_view = raft::make_device_matrix_view<IdxT, int64_t>(
//     //     refined_neighbors.data_handle(), batch.size(), top_k);
//     //   auto refined_distances_view = raft::make_device_matrix_view<T, int64_t>(
//     //     refined_distances.data_handle(), batch.size(), top_k);

//     //     // TODO: this part needs to be done in batches!!!
//     //   auto dataset_view = raft::make_device_matrix_view<const DataT, int64_t>(
//     //     dataset.data_handle(), dataset.extent(0), dataset.extent(1));
//     //   cuvs::neighbors::refine(res,
//     //                           dataset_view,
//     //                           queries_view,
//     //                           neighbor_candidates_view,
//     //                           refined_neighbors_view,
//     //                           refined_distances_view,
//     //                           index.metric);
//     //   raft::copy(h_refined_neighbors.data_handle(),
//     //              refined_neighbors_view.data_handle(),
//     //              refined_neighbors_view.size(),
//     //              raft::resource::get_cuda_stream(res));
//     //   raft::resource::sync_stream(res);

//     //   auto refined_neighbors_host_view = raft::make_host_matrix_view<int64_t, int64_t>(
//     //     h_refined_neighbors.data_handle(), batch.size(), top_k);
//     //   write_to_graph(
//     //     knn_graph, refined_neighbors_host_view, num_self_included, batch.size(),
//     batch.offset());

//   }

// }

//
// Run balanced kmeans on a subsample of the dataset to get centroids
//
template <typename T, typename IdxT = int64_t>
void get_centroids_on_data_subsample(raft::resources const& res,
                                     cuvs::distance::DistanceType metric,
                                     raft::host_matrix_view<const T, int64_t, row_major> dataset,
                                     raft::device_matrix_view<T, IdxT> centroids)
{
  size_t num_rows   = static_cast<size_t>(dataset.extent(0));
  size_t num_cols   = static_cast<size_t>(dataset.extent(1));
  size_t n_clusters = centroids.extent(0);
  size_t num_subsamples =
    std::min(static_cast<size_t>(num_rows / n_clusters), static_cast<size_t>(num_rows * 0.1));

  auto d_dataset_subsample = raft::make_device_matrix<T, int64_t>(res, num_subsamples, num_cols);
  raft::matrix::sample_rows<T, int64_t>(
    res, raft::random::RngState{0}, dataset, d_dataset_subsample.view());

  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = metric;

  cuvs::cluster::kmeans::fit(
    res, kmeans_params, raft::make_const_mdspan(d_dataset_subsample.view()), centroids);
}

template <typename T, typename IdxT = int64_t>
void get_global_nearest_clusters(
  raft::resources const& res,
  size_t num_nearest_clusters,
  size_t n_clusters,
  raft::host_matrix_view<const T, int64_t, row_major> dataset,
  raft::host_matrix_view<IdxT, IdxT, raft::row_major> global_nearest_cluster,
  raft::device_matrix_view<T, IdxT, raft::row_major> centroids,
  cuvs::distance::DistanceType metric)
{
  size_t num_rows     = static_cast<size_t>(dataset.extent(0));
  size_t num_cols     = static_cast<size_t>(dataset.extent(1));
  auto centroids_view = raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
    centroids.data_handle(), n_clusters, num_cols);

  size_t num_batches      = n_clusters;
  size_t n_rows_per_batch = (num_rows + n_clusters) / n_clusters;

  auto d_dataset_batch =
    raft::make_device_matrix<T, int64_t, raft::row_major>(res, n_rows_per_batch, num_cols);

  // this is needed because brute force search only takes int64_t type
  auto nearest_clusters_idx_int64 = raft::make_device_matrix<int64_t, int64_t, raft::row_major>(
    res, n_rows_per_batch, num_nearest_clusters);
  auto nearest_clusters_idx = raft::make_device_matrix<IdxT, int64_t, raft::row_major>(
    res, n_rows_per_batch, num_nearest_clusters);
  auto nearest_clusters_dist = raft::make_device_matrix<T, int64_t, raft::row_major>(
    res, n_rows_per_batch, num_nearest_clusters);

  // maybe do this part multi-gpu too?

  for (size_t i = 0; i < num_batches; i++) {
    //   size_t batch_size_ = batch_size;
    size_t row_offset              = n_rows_per_batch * i;
    size_t n_rows_of_current_batch = std::min(n_rows_per_batch, num_rows - row_offset);
    //   if (i == num_batches - 1) { batch_size_ = num_rows - batch_size * i; }
    raft::copy(d_dataset_batch.data_handle(),
               dataset.data_handle() + row_offset * num_cols,
               n_rows_of_current_batch * num_cols,
               resource::get_cuda_stream(res));

    std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
    cuvs::neighbors::brute_force::index<T> brute_force_index(
      res, centroids_view, norms_view, metric);
    // n_clusters is usually not large, so okay to do this brute-force
    cuvs::neighbors::brute_force::search(res,
                                         brute_force_index,
                                         raft::make_const_mdspan(d_dataset_batch.view()),
                                         nearest_clusters_idx_int64.view(),
                                         nearest_clusters_dist.view());

    thrust::copy(raft::resource::get_thrust_policy(res),
                 nearest_clusters_idx_int64.data_handle(),
                 nearest_clusters_idx_int64.data_handle() + nearest_clusters_idx_int64.size(),
                 nearest_clusters_idx.data_handle());
    raft::copy(global_nearest_cluster.data_handle() + row_offset * num_nearest_clusters,
               nearest_clusters_idx.data_handle(),
               n_rows_of_current_batch * num_nearest_clusters,
               resource::get_cuda_stream(res));
  }
}

template <typename IdxT = int64_t>
void get_inverted_indices(raft::resources const& res,
                          size_t n_clusters,
                          size_t& max_cluster_size,
                          size_t& min_cluster_size,
                          raft::host_matrix_view<IdxT, IdxT> global_nearest_cluster,
                          raft::host_vector_view<IdxT, IdxT> inverted_indices,
                          raft::host_vector_view<IdxT, IdxT> cluster_sizes,
                          raft::host_vector_view<IdxT, IdxT> cluster_offsets)
{
  // build sparse inverted indices and get number of data points for each cluster
  size_t num_rows             = global_nearest_cluster.extent(0);
  size_t num_nearest_clusters = global_nearest_cluster.extent(1);

  auto local_offsets = raft::make_host_vector<IdxT>(n_clusters);

  max_cluster_size = 0;
  min_cluster_size = std::numeric_limits<size_t>::max();

  std::fill(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters, 0);
  std::fill(local_offsets.data_handle(), local_offsets.data_handle() + n_clusters, 0);

  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_nearest_clusters; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      cluster_sizes(cluster_id) += 1;
    }
  }

  cluster_offsets(0) = 0;
  for (size_t i = 1; i < n_clusters; i++) {
    cluster_offsets(i) = cluster_offsets(i - 1) + cluster_sizes(i - 1);
  }
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_nearest_clusters; j++) {
      IdxT cluster_id = global_nearest_cluster(i, j);
      inverted_indices(cluster_offsets(cluster_id) + local_offsets(cluster_id)) = i;
      local_offsets(cluster_id) += 1;
    }
  }

  max_cluster_size = static_cast<size_t>(
    *std::max_element(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters));
  min_cluster_size = static_cast<size_t>(
    *std::min_element(cluster_sizes.data_handle(), cluster_sizes.data_handle() + n_clusters));
}

template <typename T, typename IdxT>
std::unique_ptr<batch_knn_builder<T, IdxT>> get_knn_builder(const raft::resources& handle,
                                                            batch_knn::index<IdxT, T>& index,
                                                            const index_params& params,
                                                            size_t min_cluster_size,
                                                            size_t max_cluster_size)
{
  if (index.build_algo == batch_knn::NN_DESCENT) {
    std::cout << "getting knn builder. build algo is NND\n";
    auto nn_descent_params =
      std::get<graph_build_params::nn_descent_params>(params.graph_build_params);
    return std::make_unique<batch_knn_builder_nn_descent<T, IdxT>>(handle,
                                                                   index.n_clusters,
                                                                   nn_descent_params,
                                                                   min_cluster_size,
                                                                   max_cluster_size,
                                                                   static_cast<size_t>(index.k));
  } else if (index.build_algo == batch_knn::IVF_PQ) {
    // auto ivf_pq_index_params = static_cast<ivf_pq::index_params>(index_params);
    // auto ivf_pq_search_params = static_cast<ivf_pq::search_params&>(search_params);
    auto ivf_pq_params = std::get<graph_build_params::ivf_pq_params>(params.graph_build_params);

    return std::make_unique<batch_knn_builder_ivfpq<T, IdxT>>(
      handle, index.n_clusters, ivf_pq_params);
  } else {
    RAFT_FAIL("Batch KNN build algos only supporting NN Descent and IVF PQ");
  }
}

template <typename T, typename IdxT>
void single_gpu_batch_build(const raft::resources& handle,
                            raft::host_matrix_view<const T, int64_t, row_major> dataset,
                            detail::batch_knn_builder<T, IdxT>& knn_builder,
                            IdxT* global_neighbors,
                            T* global_distances,
                            size_t max_cluster_size,
                            batch_knn::index<IdxT, T>& index,
                            const index_params& params,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                            raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  auto h_cluster_data = raft::make_host_matrix<T, int64_t, row_major>(max_cluster_size, num_cols);
  std::cout << "looking at index k here " << index.k << std::endl;
  auto batch_indices_h =
    raft::make_host_matrix<IdxT, int64_t, row_major>(max_cluster_size, index.k);
  auto batch_indices_d =
    raft::make_device_matrix<IdxT, int64_t, row_major>(handle, max_cluster_size, index.k);
  auto batch_distances_d =
    raft::make_device_matrix<float, int64_t, row_major>(handle, max_cluster_size, index.k);

  // prepare build is for large stuff
  knn_builder.prepare_build(dataset);

  for (size_t cluster_id = 0; cluster_id < index.n_clusters; cluster_id++) {
    size_t num_data_in_cluster = cluster_sizes(cluster_id);
    size_t offset              = cluster_offsets(cluster_id);

#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < num_cols; j++) {
        size_t global_row    = inverted_indices(offset + i);
        h_cluster_data(i, j) = dataset(global_row, j);
      }
    }

    // do the build now with the data.
    // if some requires device data, then do it
    knn_builder.build_knn(handle,
                          params,
                          num_data_in_cluster,
                          global_neighbors,
                          global_distances,
                          h_cluster_data.view(),
                          inverted_indices.data_handle() + offset,
                          batch_indices_h.view(),
                          batch_indices_d.view(),
                          batch_distances_d.view());
  }
}

// only supports host data for now
// use template types for index params and search params??
template <typename T, typename IdxT>
void build(const raft::resources& handle,
           raft::host_matrix_view<const T, int64_t, row_major> dataset,
           batch_knn::index<IdxT, T>& index,
           const index_params& params)  // distance type same as data type
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  auto centroids =
    raft::make_device_matrix<T, IdxT, raft::row_major>(handle, index.n_clusters, num_cols);
  get_centroids_on_data_subsample<T, IdxT>(handle, index.metric, dataset, centroids.view());

  size_t num_nearest_clusters = 2;
  auto global_nearest_cluster =
    raft::make_host_matrix<IdxT, IdxT, raft::row_major>(num_rows, num_nearest_clusters);
  get_global_nearest_clusters<T, IdxT>(handle,
                                       num_nearest_clusters,
                                       index.n_clusters,
                                       dataset,
                                       global_nearest_cluster.view(),
                                       centroids.view(),
                                       index.metric);

  auto inverted_indices =
    raft::make_host_vector<IdxT, IdxT, raft::row_major>(num_rows * num_nearest_clusters);
  auto cluster_sizes   = raft::make_host_vector<IdxT, IdxT, raft::row_major>(index.n_clusters);
  auto cluster_offsets = raft::make_host_vector<IdxT, IdxT, raft::row_major>(index.n_clusters);

  size_t max_cluster_size, min_cluster_size;
  get_inverted_indices(handle,
                       index.n_clusters,
                       max_cluster_size,
                       min_cluster_size,
                       global_nearest_cluster.view(),
                       inverted_indices.view(),
                       cluster_sizes.view(),
                       cluster_offsets.view());

  auto global_neighbors = raft::make_managed_matrix<IdxT, int64_t>(handle, num_rows, index.k);
  auto global_distances = raft::make_managed_matrix<float, int64_t>(handle, num_rows, index.k);

  std::fill(global_neighbors.data_handle(),
            global_neighbors.data_handle() + num_rows * index.k,
            std::numeric_limits<IdxT>::max());
  std::fill(global_distances.data_handle(),
            global_distances.data_handle() + num_rows * index.k,
            std::numeric_limits<float>::max());

  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);
  std::cout << "good up to this point!\n";
  std::unique_ptr<batch_knn_builder<T, IdxT>> knn_builder =
    get_knn_builder(handle, index, params, min_cluster_size, max_cluster_size);
  // auto knn_builder = get_knn_builder();
  // const raft::resources& handle,
  // batch_knn::index<IdxT, T>& index,
  // cuvs::neighbors::index_params& index_params,
  // cuvs::neighbors::index_params& search_params,
  // size_t min_cluster_size,
  // size_t max_cluster_size)

  if (clique.num_ranks_ > 1) {
    // multi gpu support
    std::cout << "multi gpu\n";
  } else {
    // single gpu support
    std::cout << "single gpu\n";
    single_gpu_batch_build(handle,
                           dataset,
                           *knn_builder,
                           global_neighbors.data_handle(),
                           global_distances.data_handle(),
                           max_cluster_size,
                           index,
                           params,
                           cluster_sizes.view(),
                           cluster_offsets.view(),
                           inverted_indices.view());
  }

  raft::copy(index.graph().data_handle(),
             global_neighbors.data_handle(),
             num_rows * index.k,
             raft::resource::get_cuda_stream(handle));
  if (index.return_distances && index.distances().has_value()) {
    raft::copy(index.distances().value().data_handle(),
               global_distances.data_handle(),
               num_rows * index.k,
               raft::resource::get_cuda_stream(handle));
  }
}

// whatever that works best
template <typename T, typename IdxT>
void build(const raft::resources& handle,
           raft::host_matrix_view<const T, int64_t, row_major> dataset,
           batch_knn::index<IdxT, T>& index)  // distance type same as data type
{
  index_params params;
  if (index.build_algo == batch_knn::NN_DESCENT) {
    // auto index_params
    auto nn_descent_index_params = nn_descent::index_params{};
    nn_descent_index_params.n_clusters =
      index.n_clusters;  // how should we move this from nn descent
    params.graph_build_params = nn_descent_index_params;
    // params.graph_build_params.n_clusters = index.n_clusters;  // how should we move this from nn
    // descent

    build(handle, dataset, index, params);
  } else if (index.build_algo == batch_knn::IVF_PQ) {
    // auto index_params  = ivf_pq::index_params{};
    // auto search_params = ivf_pq::search_params{};
    params.graph_build_params = graph_build_params::ivf_pq_params(dataset.extent, index.metric);
    build(handle, dataset, index, params);
  } else {
    // add warning that this is not supported for now
  }

  // const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);
  // // for a single node, NND may work better, and for a multi node ivfpq may work better
  // if (clique.num_ranks_ == 1) {
  //     auto index_params = nn_descent::index_params{};
  //     index_params.n_clusters = 4;    // TODO why set to 4?
  //     // add warning that we are using NND, and you might want to increase n_clusters for
  //     efficient GPU mem usage

  //     build(handle, dataset, index, index_params);
  // } else{
  //     // for multi node ivfpq may work better
  //     auto index_params = ivf_pq::index_params{};
  //     auto search_params = ivf_pq::search_params{};
  //     // add warning that we are using IVF-PQ, and you might want to increase n_clusters for
  //     efficient GPU mem usage build(handle, dataset, index, index_params, search_params);
  // }
}

// build algo defaults to NN Descent
template <typename T, typename IdxT>
batch_knn::index<IdxT, T> build(const raft::resources& handle,
                                raft::host_matrix_view<const T, int64_t, row_major> dataset,
                                int64_t k,
                                bool return_distances = false)  // distance type same as data type
{
  batch_knn::index<IdxT, T> index{handle, dataset.extent(0), k, return_distances};
  build(handle, dataset, index);
  return index;
}

// NND graph degree should be equal to k need to set this somewhere
// build algo defaults to NN Descent
template <typename T, typename IdxT = int64_t>
batch_knn::index<IdxT, T> build(const raft::resources& handle,
                                raft::host_matrix_view<const T, int64_t, row_major> dataset,
                                int64_t k,
                                const index_params& params,
                                bool return_distances = false)  // distance type same as data type
{
  std::cout << "k is " << k << std::endl;
  batch_knn::index<IdxT, T> index{handle, dataset.extent(0), k, false};
  build(handle, dataset, index, params);
  return index;
}

// // build algo defaults to NN Descent
// template <typename T, typename IdxT=int64_t>
// batch_knn::index<IdxT, T> build(const raft::resources& handle)   // distance type same as data
// type
// {
//     batch_knn::index<IdxT, T> index{handle, 30, 10, false};
//     // build(handle, dataset, index, index_params, search_params);
//     return index;
// }

}  // namespace cuvs::neighbors::batch_knn::detail
