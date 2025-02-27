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
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/resource/nccl_clique.hpp>

#include <cuvs/cluster/kmeans.hpp>
#include <raft/matrix/sample_rows.cuh>

#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/util/cudart_utils.hpp>
#include <variant>

#include "batch_knn_builder.cuh"
#include <cuvs/neighbors/batch_knn.hpp>

namespace cuvs::neighbors::batch_knn::detail {
using namespace cuvs::neighbors;

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

  auto dataset_subsample_d = raft::make_device_matrix<T, int64_t>(res, num_subsamples, num_cols);
  raft::matrix::sample_rows<T, int64_t>(
    res, raft::random::RngState{0}, dataset, dataset_subsample_d.view());

  cuvs::cluster::kmeans::balanced_params kmeans_params;
  kmeans_params.metric = metric;

  cuvs::cluster::kmeans::fit(
    res, kmeans_params, raft::make_const_mdspan(dataset_subsample_d.view()), centroids);
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

  auto dataset_batch_d =
    raft::make_device_matrix<T, int64_t, raft::row_major>(res, n_rows_per_batch, num_cols);

  // this is needed because brute force search only takes int64_t type
  auto nearest_clusters_idx_int64_d = raft::make_device_matrix<int64_t, int64_t, raft::row_major>(
    res, n_rows_per_batch, num_nearest_clusters);
  auto nearest_clusters_idx_d = raft::make_device_matrix<IdxT, int64_t, raft::row_major>(
    res, n_rows_per_batch, num_nearest_clusters);
  auto nearest_clusters_dist_d = raft::make_device_matrix<T, int64_t, raft::row_major>(
    res, n_rows_per_batch, num_nearest_clusters);

  // maybe do this part multi-gpu too?
  for (size_t i = 0; i < num_batches; i++) {
    size_t row_offset              = n_rows_per_batch * i;
    size_t n_rows_of_current_batch = std::min(n_rows_per_batch, num_rows - row_offset);
    raft::copy(dataset_batch_d.data_handle(),
               dataset.data_handle() + row_offset * num_cols,
               n_rows_of_current_batch * num_cols,
               resource::get_cuda_stream(res));

    std::optional<raft::device_vector_view<const T, int64_t>> norms_view;
    cuvs::neighbors::brute_force::index<T> brute_force_index(
      res, centroids_view, norms_view, metric);

    // n_clusters is usually not large, so okay to do this brute-force
    cuvs::neighbors::brute_force::search(res,
                                         brute_force_index,
                                         raft::make_const_mdspan(dataset_batch_d.view()),
                                         nearest_clusters_idx_int64_d.view(),
                                         nearest_clusters_dist_d.view());

    thrust::copy(raft::resource::get_thrust_policy(res),
                 nearest_clusters_idx_int64_d.data_handle(),
                 nearest_clusters_idx_int64_d.data_handle() + nearest_clusters_idx_int64_d.size(),
                 nearest_clusters_idx_d.data_handle());
    raft::copy(global_nearest_cluster.data_handle() + row_offset * num_nearest_clusters,
               nearest_clusters_idx_d.data_handle(),
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
  if (std::holds_alternative<graph_build_params::nn_descent_params>(params.graph_build_params)) {
    auto nn_descent_params =
      std::get<graph_build_params::nn_descent_params>(params.graph_build_params);

    return std::make_unique<batch_knn_builder_nn_descent<T, IdxT>>(handle,
                                                                   params.n_clusters,
                                                                   nn_descent_params,
                                                                   min_cluster_size,
                                                                   max_cluster_size,
                                                                   static_cast<size_t>(index.k),
                                                                   index.return_distances);
  } else if (std::holds_alternative<graph_build_params::ivf_pq_params>(params.graph_build_params)) {
    auto ivf_pq_params = std::get<graph_build_params::ivf_pq_params>(params.graph_build_params);
    return std::make_unique<batch_knn_builder_ivfpq<T, IdxT>>(handle,
                                                              params.n_clusters,
                                                              ivf_pq_params,
                                                              min_cluster_size,
                                                              max_cluster_size,
                                                              static_cast<size_t>(index.k));
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
                            size_t n_clusters,
                            batch_knn::index<IdxT, T>& index,
                            const index_params& batch_params,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                            raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                            raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  auto cluster_data = raft::make_host_matrix<T, int64_t, row_major>(max_cluster_size, num_cols);

  auto batch_indices_h =
    raft::make_host_matrix<IdxT, int64_t, row_major>(max_cluster_size, index.k);
  auto batch_indices_d =
    raft::make_device_matrix<IdxT, int64_t, row_major>(handle, max_cluster_size, index.k);
  auto batch_distances_d =
    raft::make_device_matrix<float, int64_t, row_major>(handle, max_cluster_size, index.k);

  knn_builder.prepare_build(dataset);
  raft::print_host_vector("cluster sizes", cluster_sizes.data_handle(), n_clusters, std::cout);

  for (size_t cluster_id = 0; cluster_id < n_clusters; cluster_id++) {
    printf("=============== [THREAD %lu] Cluster [%lu / %lu] ===============\n",
           static_cast<size_t>(omp_get_thread_num()),
           cluster_id + 1,
           n_clusters);
    size_t num_data_in_cluster = cluster_sizes(cluster_id);
    size_t offset              = cluster_offsets(cluster_id);
    // printf("\t[THREAD %lu] num data in cluster %lu, offset %lu\n",
    //        static_cast<size_t>(omp_get_thread_num()),
    //        num_data_in_cluster,
    //        offset);
#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < num_cols; j++) {
        size_t global_row  = inverted_indices(offset + i);
        cluster_data(i, j) = dataset(global_row, j);
      }
    }

    auto cluster_data_view = raft::make_host_matrix_view<const T, int64_t>(
      cluster_data.data_handle(), num_data_in_cluster, num_cols);
    auto inverted_indices_view = raft::make_host_vector_view<IdxT, int64_t>(
      inverted_indices.data_handle() + offset, num_data_in_cluster);

    knn_builder.build_knn(handle,
                          batch_params,
                          num_data_in_cluster,
                          global_neighbors,
                          global_distances,
                          cluster_data_view,
                          inverted_indices_view,
                          batch_indices_h.view(),
                          batch_indices_d.view(),
                          batch_distances_d.view());
  }
}

template <typename T, typename IdxT>
void multi_gpu_batch_build(const raft::resources& handle,
                           raft::host_matrix_view<const T, int64_t, row_major> dataset,
                           IdxT* global_neighbors,
                           T* global_distances,
                           size_t max_cluster_size,
                           size_t min_cluster_size,
                           batch_knn::index<IdxT, T>& index,
                           const index_params& params,
                           raft::host_vector_view<IdxT, IdxT, row_major> cluster_sizes,
                           raft::host_vector_view<IdxT, IdxT, row_major> cluster_offsets,
                           raft::host_vector_view<IdxT, IdxT, row_major> inverted_indices)
{
  size_t num_rows = dataset.extent(0);
  size_t num_cols = dataset.extent(1);

  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);
  size_t clusters_per_rank               = params.n_clusters / clique.num_ranks_;
  size_t rem = params.n_clusters - clusters_per_rank * clique.num_ranks_;

  // std::unique_ptr<batch_knn_builder<T, IdxT>> new_knn_builder =
  //   get_knn_builder(handle, index, params, min_cluster_size, max_cluster_size);

#pragma omp parallel for
  for (int rank = 0; rank < clique.num_ranks_; rank++) {
    int dev_id                            = clique.device_ids_[rank];
    const raft::device_resources& dev_res = clique.device_resources_[rank];
    RAFT_CUDA_TRY(cudaSetDevice(dev_id));

    std::unique_ptr<batch_knn_builder<T, IdxT>> knn_builder =
      get_knn_builder(dev_res, index, params, min_cluster_size, max_cluster_size);

    size_t num_data_for_this_rank = 0;
    size_t base_cluster_idx       = rank * clusters_per_rank + std::min((size_t)rank, rem);
    size_t num_clusters_for_this_rank =
      (size_t)rank < rem
        ? clusters_per_rank + 1
        : clusters_per_rank;  // std::min(clusters_per_rank, params.n_clusters - base_cluster_idx);
    for (size_t p = 0; p < num_clusters_for_this_rank; p++) {
      num_data_for_this_rank += cluster_sizes(base_cluster_idx + p);
    }

    size_t rank_offset = cluster_offsets(base_cluster_idx);
    // printf(
    //   "[THREAD %lu] num data for this rank %lu, base cluster idx %lu, num clusters for this rank
    //   "
    //   "%lu, rank offset: %lu\n",
    //   static_cast<size_t>(omp_get_thread_num()),
    //   num_data_for_this_rank,
    //   base_cluster_idx,
    //   num_clusters_for_this_rank,
    //   rank_offset);
    // re-map offsets for each rank
    for (size_t p = 0; p < num_clusters_for_this_rank; p++) {
      cluster_offsets(base_cluster_idx + p) -= rank_offset;
    }

    // TODO: maybe find a new max cluster size per rank for more GPU mem efficiency

    auto cluster_sizes_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      cluster_sizes.data_handle() + base_cluster_idx, num_clusters_for_this_rank);
    auto cluster_offsets_for_this_rank = raft::make_host_vector_view<IdxT, IdxT>(
      cluster_offsets.data_handle() + base_cluster_idx, num_clusters_for_this_rank);
    auto cluster_inverted_indices = raft::make_host_vector_view<IdxT, IdxT>(
      inverted_indices.data_handle() + rank_offset, num_clusters_for_this_rank);
    single_gpu_batch_build(dev_res,
                           dataset,
                           *knn_builder,
                           global_neighbors,
                           global_distances,
                           max_cluster_size,
                           num_clusters_for_this_rank,
                           index,
                           params,
                           cluster_sizes_for_this_rank,
                           cluster_offsets_for_this_rank,
                           cluster_inverted_indices);
  }
}

// TODO: moving

// only supports host data for now
template <typename T, typename IdxT>
void build(const raft::resources& handle,
           raft::host_matrix_view<const T, int64_t, row_major> dataset,
           const index_params& batch_params,
           batch_knn::index<IdxT, T>& index)  // distance type same as data type
{
  size_t num_rows = static_cast<size_t>(dataset.extent(0));
  size_t num_cols = static_cast<size_t>(dataset.extent(1));

  size_t num_nearest_clusters = batch_params.num_nearest_clusters;
  size_t n_clusters           = batch_params.n_clusters;

  auto centroids = raft::make_device_matrix<T, IdxT, raft::row_major>(handle, n_clusters, num_cols);
  get_centroids_on_data_subsample<T, IdxT>(handle, batch_params.metric, dataset, centroids.view());

  auto global_nearest_cluster =
    raft::make_host_matrix<IdxT, IdxT, raft::row_major>(num_rows, num_nearest_clusters);
  get_global_nearest_clusters<T, IdxT>(handle,
                                       num_nearest_clusters,
                                       n_clusters,
                                       dataset,
                                       global_nearest_cluster.view(),
                                       centroids.view(),
                                       batch_params.metric);

  auto inverted_indices =
    raft::make_host_vector<IdxT, IdxT, raft::row_major>(num_rows * num_nearest_clusters);
  auto cluster_sizes   = raft::make_host_vector<IdxT, IdxT, raft::row_major>(n_clusters);
  auto cluster_offsets = raft::make_host_vector<IdxT, IdxT, raft::row_major>(n_clusters);

  size_t max_cluster_size, min_cluster_size;
  get_inverted_indices(handle,
                       n_clusters,
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

  // std::unique_ptr<batch_knn_builder<T, IdxT>> knn_builder =
  //   get_knn_builder(handle, index, batch_params, min_cluster_size, max_cluster_size);

  const raft::comms::nccl_clique& clique = raft::resource::get_nccl_clique(handle);

  if (clique.num_ranks_ > 1) {
    // multi gpu support
    std::cout << "Running multi gpu\n";
    multi_gpu_batch_build(handle,
                          dataset,
                          global_neighbors.data_handle(),
                          global_distances.data_handle(),
                          max_cluster_size,
                          min_cluster_size,
                          index,
                          batch_params,
                          cluster_sizes.view(),
                          cluster_offsets.view(),
                          inverted_indices.view());
  } else {
    // single gpu support
    std::cout << "Running single gpu\n";
    std::unique_ptr<batch_knn_builder<T, IdxT>> knn_builder =
      get_knn_builder(handle, index, batch_params, min_cluster_size, max_cluster_size);
    single_gpu_batch_build(handle,
                           dataset,
                           *knn_builder,
                           global_neighbors.data_handle(),
                           global_distances.data_handle(),
                           max_cluster_size,
                           n_clusters,
                           index,
                           batch_params,
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

// defaults to NND
template <typename T, typename IdxT>
void build(const raft::resources& handle,
           raft::host_matrix_view<const T, int64_t, row_major> dataset,
           batch_knn::index<IdxT, T>& index)  // distance type same as data type
{
  index_params batch_params;
  build(handle, dataset, batch_params, index);
}

// NND graph degree should be equal to k need to set this somewhere
// build algo defaults to NN Descent
template <typename T, typename IdxT = int64_t>
batch_knn::index<IdxT, T> build(const raft::resources& handle,
                                raft::host_matrix_view<const T, int64_t, row_major> dataset,
                                int64_t k,
                                const index_params& batch_params,
                                bool return_distances = false)  // distance type same as data type
{
  batch_knn::index<IdxT, T> index{handle, dataset.extent(0), k, return_distances};
  build(handle, dataset, batch_params, index);
  return index;
}

// whatever that works best
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

}  // namespace cuvs::neighbors::batch_knn::detail
