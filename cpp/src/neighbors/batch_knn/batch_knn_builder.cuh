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
#include <cstddef>
#include <cuda.h>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/util/cudart_utils.hpp>
//  #include "batch_knn_common.cuh"
#include "cuvs/neighbors/batch_knn.hpp"
#include "cuvs/neighbors/refine.hpp"

namespace cuvs::neighbors::batch_knn::detail {
using namespace cuvs::neighbors;
using align32 = raft::Pow2<32>;

template <typename KeyType, typename ValueType>
struct KeyValuePair {
  KeyType key;
  ValueType value;
};

template <typename KeyType, typename ValueType>
struct CustomKeyComparator {
  __device__ bool operator()(const KeyValuePair<KeyType, ValueType>& a,
                             const KeyValuePair<KeyType, ValueType>& b) const
  {
    if (a.key == b.key) { return a.value < b.value; }
    return a.key < b.key;
  }
};

template <typename IdxT, int BLOCK_SIZE, int ITEMS_PER_THREAD>
RAFT_KERNEL merge_subgraphs_kernel(IdxT* cluster_data_indices,
                                   size_t graph_degree,
                                   size_t num_cluster_in_batch,
                                   float* global_distances,
                                   float* batch_distances,
                                   IdxT* global_indices,
                                   IdxT* batch_indices)
{
  size_t batch_row = blockIdx.x;
  typedef cub::BlockMergeSort<KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>
    BlockMergeSortType;
  __shared__ typename cub::BlockMergeSort<KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>::
    TempStorage tmpSmem;

  extern __shared__ char sharedMem[];
  float* blockKeys  = reinterpret_cast<float*>(sharedMem);
  IdxT* blockValues = reinterpret_cast<IdxT*>(&sharedMem[graph_degree * 2 * sizeof(float)]);
  int16_t* uniqueMask =
    reinterpret_cast<int16_t*>(&sharedMem[graph_degree * 2 * (sizeof(float) + sizeof(IdxT))]);

  if (batch_row < num_cluster_in_batch) {
    // load batch or global depending on threadIdx
    size_t global_row = cluster_data_indices[batch_row];

    KeyValuePair<float, IdxT> threadKeyValuePair[ITEMS_PER_THREAD];

    size_t halfway   = BLOCK_SIZE / 2;
    size_t do_global = threadIdx.x < halfway;

    float* distances;
    IdxT* indices;

    if (do_global) {
      distances = global_distances;
      indices   = global_indices;
    } else {
      distances = batch_distances;
      indices   = batch_indices;
    }

    size_t idxBase = (threadIdx.x * do_global + (threadIdx.x - halfway) * (1lu - do_global)) *
                     static_cast<size_t>(ITEMS_PER_THREAD);
    size_t arrIdxBase = (global_row * do_global + batch_row * (1lu - do_global)) * graph_degree;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < graph_degree) {
        threadKeyValuePair[i].key   = distances[arrIdxBase + colId];
        threadKeyValuePair[i].value = indices[arrIdxBase + colId];
      } else {
        threadKeyValuePair[i].key   = std::numeric_limits<float>::max();
        threadKeyValuePair[i].value = std::numeric_limits<IdxT>::max();
      }
    }

    __syncthreads();

    BlockMergeSortType(tmpSmem).Sort(threadKeyValuePair, CustomKeyComparator<float, IdxT>{});

    // load sorted result into shared memory to get unique values
    idxBase = threadIdx.x * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < 2 * graph_degree) {
        blockKeys[colId]   = threadKeyValuePair[i].key;
        blockValues[colId] = threadKeyValuePair[i].value;
      }
    }

    __syncthreads();

    // get unique mask
    if (threadIdx.x == 0) { uniqueMask[0] = 1; }
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        uniqueMask[colId] = static_cast<int16_t>(blockValues[colId] != blockValues[colId - 1]);
      }
    }

    __syncthreads();

    // prefix sum
    if (threadIdx.x == 0) {
      for (int i = 1; i < 2 * graph_degree; i++) {
        uniqueMask[i] += uniqueMask[i - 1];
      }
    }

    __syncthreads();
    // load unique values to global memory
    if (threadIdx.x == 0) {
      global_distances[global_row * graph_degree] = blockKeys[0];
      global_indices[global_row * graph_degree]   = blockValues[0];
    }

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        bool is_unique       = uniqueMask[colId] != uniqueMask[colId - 1];
        int16_t global_colId = uniqueMask[colId] - 1;
        if (is_unique && static_cast<size_t>(global_colId) < graph_degree) {
          global_distances[global_row * graph_degree + global_colId] = blockKeys[colId];
          global_indices[global_row * graph_degree + global_colId]   = blockValues[colId];
        }
      }
    }
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            host_device_accessor<std::experimental::default_accessor<T>, memory_type::host>>
void merge_subgraphs(raft::resources const& res,
                     size_t k,
                     size_t num_data_in_cluster,
                     IdxT* inverted_indices_d,
                     T* global_distances,
                     T* batch_distances_d,
                     IdxT* global_neighbors,
                     IdxT* batch_indices_d)
{
  size_t num_elems     = k * 2;
  size_t sharedMemSize = num_elems * (sizeof(float) + sizeof(IdxT) + sizeof(int16_t));
  if (num_elems <= 128) {
    merge_subgraphs_kernel<IdxT, 32, 4>
      <<<num_data_in_cluster, 32, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        inverted_indices_d,
        k,
        num_data_in_cluster,
        global_distances,
        batch_distances_d,
        global_neighbors,
        batch_indices_d);
  } else if (num_elems <= 512) {
    merge_subgraphs_kernel<IdxT, 128, 4>
      <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        inverted_indices_d,
        k,
        num_data_in_cluster,
        global_distances,
        batch_distances_d,
        global_neighbors,
        batch_indices_d);
  } else if (num_elems <= 1024) {
    merge_subgraphs_kernel<IdxT, 128, 8>
      <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        inverted_indices_d,
        k,
        num_data_in_cluster,
        global_distances,
        batch_distances_d,
        global_neighbors,
        batch_indices_d);
  } else if (num_elems <= 2048) {
    merge_subgraphs_kernel<IdxT, 256, 8>
      <<<num_data_in_cluster, 256, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
        inverted_indices_d,
        k,
        num_data_in_cluster,
        global_distances,
        batch_distances_d,
        global_neighbors,
        batch_indices_d);
  } else {
    // this is as far as we can get due to the shared mem usage of cub::BlockMergeSort
    RAFT_FAIL("The degree of knn is too large (%lu). It must be smaller than 1024", k);
  }
  raft::resource::sync_stream(res);
}

template <typename T, typename IdxT = int64_t>
struct batch_knn_builder {
  batch_knn_builder() {}

  virtual void prepare_build(raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    // preparing build (index of making gnnd etc)
  }

  virtual void build_knn(raft::resources const& res,
                         const index_params& params,
                         size_t num_data_in_cluster,
                         IdxT* global_neighbors,
                         T* global_distances,
                         raft::host_matrix_view<const T, int64_t, row_major> dataset,
                         IdxT* inverted_indices,
                         raft::host_matrix_view<IdxT, IdxT, row_major> batch_indices_h,
                         raft::device_matrix_view<IdxT, int64_t, row_major> batch_indices_d,
                         raft::device_matrix_view<T, int64_t, row_major> batch_distances_d)
  {
    // build for nnd, search and refinement for ivfpq
  }
};

template <typename T, typename IdxT = int64_t>
struct batch_knn_builder_ivfpq : public batch_knn_builder<T, IdxT> {
  batch_knn_builder_ivfpq(raft::resources const& res,
                          size_t n_clusters,
                          batch_knn::graph_build_params::ivf_pq_params& params,
                          size_t min_cluster_size,
                          size_t max_cluster_size,
                          size_t k)
    : batch_knn_builder<T, IdxT>(),
      res{res},
      index_params{params.build_params},
      search_params{params.search_params},
      k{k},
      n_clusters{n_clusters},
      min_cluster_size{min_cluster_size},
      max_cluster_size{max_cluster_size}
  {
    refinement_rate                = 3.0;
    index_params.add_data_on_build = false;
    // take care of this part
    // index_params.kmeans_trainset_fraction     = 1.0;  // what percentage of data do you want to
    // use
    index_params.kmeans_n_iters               = 50;
    index_params.n_lists                      = 20;
    index_params.max_train_points_per_pq_code = 512;
  }

  void prepare_build(raft::host_matrix_view<const T, int64_t, row_major> dataset) override
  {
    // build ivf-pq index on a random subset for efficient GPU memory usage

    std::cout << "index params add dat aon build " << index_params.add_data_on_build
              << "  n clusters " << n_clusters << " n lists " << index_params.n_lists << std::endl;
    size_t num_rows = static_cast<size_t>(dataset.extent(0));
    size_t num_cols = static_cast<size_t>(dataset.extent(1));

    size_t num_subsamples =
      std::min(static_cast<size_t>(num_rows / n_clusters), static_cast<size_t>(num_rows * 0.1));
    index_params.kmeans_trainset_fraction = (float)num_subsamples / (float)num_rows;

    std::cout << "number of subsamples " << num_subsamples
              << "fraction: " << index_params.kmeans_trainset_fraction << std::endl;
    // auto d_dataset_subsample =
    //   raft::make_device_matrix<T, int64_t, row_major>(res, num_subsamples, num_cols);
    // raft::matrix::sample_rows<T, int64_t>(
    //   res, raft::random::RngState{0}, dataset, d_dataset_subsample.view());

    // auto d_dataset_subsample =
    //   raft::make_device_matrix<T, int64_t, row_major>(res, num_rows, num_cols);
    // raft::copy(d_dataset_subsample.data_handle(),
    //            dataset.data_handle(),
    //            num_rows * num_cols,
    //            raft::resource::get_cuda_stream(res));

    // auto index_empty = ivf_pq::build(res, index_params, dataset);

    index.emplace(ivf_pq::build(res, index_params, dataset));
    std::cout << "index codebook size: this should be empty" << index.value().size() << std::endl;

    // extending with the full dataset
    for (size_t i = 0; i < n_clusters; i++) {}
    cuvs::neighbors::ivf_pq::extend(res, dataset, std::nullopt, &index.value());

    // index.emplace(cuvs::neighbors::ivf_pq::build(
    //   res, index_params, raft::make_const_mdspan(d_dataset_subsample.view())));

    std::cout << "index codebook size after extend with full data " << index.value().size()
              << std::endl;
    size_t top_k     = k + 1;
    size_t gpu_top_k = k * refinement_rate;
    // size_t gpu_top_k = num_rows - 1;
    gpu_top_k = std::min<IdxT>(std::max(gpu_top_k, top_k), min_cluster_size);
    std::cout << "distance of index if ivfpq " << index.value().metric() << "gput top k"
              << gpu_top_k << std::endl;
    queries_d.emplace(
      raft::make_device_matrix<T, int64_t, row_major>(res, max_cluster_size, num_cols));
    data_d.emplace(
      raft::make_device_matrix<T, int64_t, row_major>(res, max_cluster_size, num_cols));
    distances_candidate_d.emplace(
      raft::make_device_matrix<T, int64_t, row_major>(res, max_cluster_size, gpu_top_k));
    neighbors_candidate_d.emplace(
      raft::make_device_matrix<IdxT, int64_t, row_major>(res, max_cluster_size, gpu_top_k));
    inverted_indices_d.emplace(raft::make_device_vector<IdxT, IdxT>(res, max_cluster_size));
    // we might have to send this to multiple GPUs if multi-gpu setting
  }

  void build_knn(raft::resources const& res,
                 const index_params& params,
                 size_t num_data_in_cluster,
                 IdxT* global_neighbors,
                 T* global_distances,
                 raft::host_matrix_view<const T, int64_t, row_major> dataset,
                 IdxT* inverted_indices,
                 raft::host_matrix_view<IdxT, IdxT, row_major> batch_indices_h,
                 raft::device_matrix_view<IdxT, int64_t, row_major> batch_indices_d,
                 raft::device_matrix_view<T, int64_t, row_major> batch_distances_d) override
  {
    // build for nnd, search and refinement for ivfpq
    std::cout << "HERE index codebook size " << index.value().pq_book_size() << std::endl;
    std::cout << "attempting to build knn with overrode function in ivfpq builder\n";
    raft::print_host_vector("inverted indices", inverted_indices, 20, std::cout);
    size_t num_cols = dataset.extent(1);
    raft::copy(queries_d.value().data_handle(),
               dataset.data_handle(),
               num_data_in_cluster * num_cols,
               raft::resource::get_cuda_stream(res));
    raft::copy(data_d.value().data_handle(),
               queries_d.value().data_handle(),
               num_data_in_cluster * num_cols,
               raft::resource::get_cuda_stream(res));

    size_t top_k     = k + 1;
    size_t gpu_top_k = k * refinement_rate;
    gpu_top_k        = std::min<IdxT>(std::max(gpu_top_k, top_k), min_cluster_size);

    auto queries_view = raft::make_device_matrix_view<const T, int64_t>(
      queries_d.value().data_handle(), num_data_in_cluster, dataset.extent(1));
    auto distances_candidate_view = raft::make_device_matrix_view<T, int64_t>(
      distances_candidate_d.value().data_handle(), num_data_in_cluster, gpu_top_k);
    auto neighbors_candidate_view = raft::make_device_matrix_view<IdxT, int64_t>(
      neighbors_candidate_d.value().data_handle(), num_data_in_cluster, gpu_top_k);
    std::cout << " k is: " << k << " topk " << top_k << " gpu top k " << gpu_top_k
              << " num data in cluster: " << num_data_in_cluster << std::endl;

    raft::print_device_vector("queries for search", queries_view.data_handle(), 10, std::cout);
    raft::print_device_vector(
      "queries for search", queries_view.data_handle() + dataset.extent(1), 10, std::cout);
    cuvs::neighbors::ivf_pq::search(res,
                                    search_params,
                                    index.value(),
                                    queries_view,
                                    neighbors_candidate_view,
                                    distances_candidate_view);
    raft::print_device_vector(
      "neighbors candidate view 0", neighbors_candidate_d.value().data_handle(), 10, std::cout);
    raft::print_device_vector("neighbors candidate view 1",
                              neighbors_candidate_d.value().data_handle() + gpu_top_k,
                              10,
                              std::cout);
    raft::print_device_vector(
      "distances candidate view 0", distances_candidate_d.value().data_handle(), 10, std::cout);
    raft::print_device_vector("distances candidate view 1",
                              distances_candidate_d.value().data_handle() + gpu_top_k,
                              10,
                              std::cout);
    auto resulting_indices_d = raft::make_device_matrix_view<IdxT, int64_t>(
      batch_indices_d.data_handle(), num_data_in_cluster, k);
    auto resulting_distances_d = raft::make_device_matrix_view<T, int64_t>(
      batch_distances_d.data_handle(), num_data_in_cluster, k);

    auto data_view =
      raft::make_device_matrix_view(data_d.value().data_handle(), num_data_in_cluster, num_cols);
    refine(res,
           raft::make_const_mdspan(queries_view),
           raft::make_const_mdspan(data_view),
           raft::make_const_mdspan(neighbors_candidate_view),
           resulting_indices_d,
           resulting_distances_d);  // TODO: define metric here too
    // TODO: refine includes itself

    raft::print_device_vector(
      "AFTER REFINEMENT neighbors 0", resulting_indices_d.data_handle(), 10, std::cout);
    raft::print_device_vector(
      "AFTER REFINEMENT neighbors 1", resulting_indices_d.data_handle() + k, 10, std::cout);
    raft::print_device_vector(
      "AFTER REFINEMENT distances 0", resulting_distances_d.data_handle(), 10, std::cout);
    raft::print_device_vector(
      "AFTER REFINEMENT distances 1", resulting_distances_d.data_handle() + k, 10, std::cout);

    auto tmp_indices_h = raft::make_host_matrix<IdxT, int64_t>(num_data_in_cluster, k);
    raft::copy(tmp_indices_h.data_handle(),
               resulting_indices_d.data_handle(),
               num_data_in_cluster * k,
               raft::resource::get_cuda_stream(res));

    // remap indices
#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < k; j++) {
        size_t local_idx      = tmp_indices_h(i, j);
        batch_indices_h(i, j) = inverted_indices[local_idx];
      }
    }

    raft::print_host_vector(
      "remapped batch indices 0", batch_indices_h.data_handle(), 10, std::cout);
    raft::print_host_vector(
      "remapped batch indices 1", batch_indices_h.data_handle() + k, 10, std::cout);

    raft::copy(inverted_indices_d.value().data_handle(),
               inverted_indices,
               num_data_in_cluster,
               raft::resource::get_cuda_stream(res));

    raft::copy(batch_indices_d.data_handle(),
               batch_indices_h.data_handle(),
               num_data_in_cluster * k,
               raft::resource::get_cuda_stream(res));

    merge_subgraphs(res,
                    k,
                    num_data_in_cluster,
                    inverted_indices_d.value().data_handle(),
                    global_distances,
                    batch_distances_d.data_handle(),
                    global_neighbors,
                    batch_indices_d.data_handle());

    // we need to ensure the copy operations are done prior using the host data
    raft::resource::sync_stream(res);
  }

  // TODO: index doesn't need init?
  ivf_pq::index_params index_params;
  ivf_pq::search_params search_params;
  float refinement_rate;
  std::optional<ivf_pq::index<IdxT>> index;
  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size, k;
  // int64_t kMaxQueries = 4096;

  std::optional<raft::device_matrix<T, int64_t>> queries_d;
  std::optional<raft::device_matrix<T, int64_t>> data_d;
  std::optional<raft::device_matrix<T, int64_t>> distances_candidate_d;
  std::optional<raft::device_matrix<IdxT, int64_t>> neighbors_candidate_d;
  std::optional<raft::device_vector<IdxT, int64_t>> inverted_indices_d;
};

template <typename T, typename IdxT = int64_t>
struct batch_knn_builder_nn_descent : public batch_knn_builder<T, IdxT> {
  batch_knn_builder_nn_descent(raft::resources const& res,
                               size_t n_clusters,
                               batch_knn::graph_build_params::nn_descent_params& index_params,
                               size_t min_cluster_size,
                               size_t max_cluster_size,
                               size_t k)
    : batch_knn_builder<T, IdxT>(),
      res{res},
      k{k},
      n_clusters{n_clusters},
      min_cluster_size{min_cluster_size},
      max_cluster_size{max_cluster_size}
  {
    index_params = index_params;

    std::cout << "max slucter size: " << max_cluster_size << " min: " << min_cluster_size
              << " k : " << k << std::endl;
    // make int graph
  }

  void prepare_build(raft::host_matrix_view<const T, int64_t, row_major> dataset) override
  {
    size_t intermediate_degree = index_params.intermediate_graph_degree;
    size_t graph_degree        = k;

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
    int_graph.emplace(raft::make_host_matrix<int, int64_t, row_major>(
      max_cluster_size, static_cast<int64_t>(extended_graph_degree)));
    inverted_indices_d.emplace(raft::make_device_vector<IdxT, IdxT>(res, max_cluster_size));
  }

  void build_knn(raft::resources const& res,
                 const index_params& params,
                 size_t num_data_in_cluster,
                 IdxT* global_neighbors,
                 T* global_distances,
                 raft::host_matrix_view<const T, int64_t, row_major> dataset,
                 IdxT* inverted_indices,
                 raft::host_matrix_view<IdxT, IdxT, row_major> batch_indices_h,
                 raft::device_matrix_view<IdxT, int64_t, row_major> batch_indices_d,
                 raft::device_matrix_view<T, int64_t, row_major> batch_distances_d) override
  {
    // build for nnd, search and refinement for ivfpq
    if (nnd_builder.has_value()) {
      auto int_graph_ptr = int_graph.value().data_handle();
      nnd_builder.value().build(dataset.data_handle(),
                                (int)num_data_in_cluster,
                                int_graph_ptr,
                                true,
                                batch_distances_d.data_handle());
    }

    // remap indices
#pragma omp parallel for
    for (size_t i = 0; i < num_data_in_cluster; i++) {
      for (size_t j = 0; j < k; j++) {
        size_t local_idx      = int_graph.value()(i, j);
        batch_indices_h(i, j) = inverted_indices[local_idx];
      }
    }

    raft::copy(inverted_indices_d.value().data_handle(),
               inverted_indices,
               num_data_in_cluster,
               raft::resource::get_cuda_stream(res));

    raft::copy(batch_indices_d.data_handle(),
               batch_indices_h.data_handle(),
               num_data_in_cluster * k,
               raft::resource::get_cuda_stream(res));

    merge_subgraphs(res,
                    k,
                    num_data_in_cluster,
                    inverted_indices_d.value().data_handle(),
                    global_distances,
                    batch_distances_d.data_handle(),
                    global_neighbors,
                    batch_indices_d.data_handle());
  }

  raft::resources const& res;
  size_t n_clusters, min_cluster_size, max_cluster_size, k;
  nn_descent::index_params index_params;
  nn_descent::detail::BuildConfig build_config;

  std::optional<nn_descent::detail::GNND<const T, int>> nnd_builder;
  std::optional<raft::host_matrix<int, int64_t, row_major>> int_graph;
  std::optional<raft::device_vector<IdxT, int64_t>> inverted_indices_d;
  // nn_descent::detail::GNND<const T, int> nnd;
};

}  // namespace cuvs::neighbors::batch_knn::detail
