
/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "ann_utils.cuh"
#include "cagra/device_common.hpp"
#include "cuvs/distance/distance.h"
#include "nn_descent_gnnd.hpp"

#include <cuda_runtime_api.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/util/arch.cuh>  // raft::util::arch::SM_*
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <mma.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <type_traits>

namespace cuvs::neighbors::nn_descent::detail {

template <typename Index_t>
struct ResultItem;

template <>
class ResultItem<int> {
 private:
  using Index_t = int;
  Index_t id_;
  DistData_t dist_;

 public:
  __host__ __device__ ResultItem()
    : id_(std::numeric_limits<Index_t>::max()), dist_(std::numeric_limits<DistData_t>::max()) {};
  __host__ __device__ ResultItem(const Index_t id_with_flag, const DistData_t dist)
    : id_(id_with_flag), dist_(dist) {};
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t& id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  __host__ __device__ DistData_t& dist() { return dist_; }

  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }

  __host__ __device__ bool operator<(const ResultItem<Index_t>& other) const
  {
    if (dist_ == other.dist_) return id() < other.id();
    return dist_ < other.dist_;
  }
  __host__ __device__ bool operator==(const ResultItem<Index_t>& other) const
  {
    return id() == other.id();
  }
  __host__ __device__ bool operator>=(const ResultItem<Index_t>& other) const
  {
    return !(*this < other);
  }
  __host__ __device__ bool operator<=(const ResultItem<Index_t>& other) const
  {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ bool operator>(const ResultItem<Index_t>& other) const
  {
    return !(*this <= other);
  }
  __host__ __device__ bool operator!=(const ResultItem<Index_t>& other) const
  {
    return !(*this == other);
  }
};

using align32 = raft::Pow2<32>;

template <typename T>
int get_batch_size(const int it_now, const T nrow, const int batch_size)
{
  int it_total = raft::ceildiv(nrow, batch_size);
  return (it_now == it_total - 1) ? nrow - it_now * batch_size : batch_size;
}

// for avoiding bank conflict
template <typename T>
constexpr __host__ __device__ __forceinline__ int skew_dim(int ndim)
{
  // all "4"s are for alignment
  if constexpr (std::is_same<T, float>::value) {
    ndim = raft::ceildiv(ndim, 4) * 4;
    return ndim + (ndim % 32 == 0) * 4;
  }
}

template <typename T>
__device__ __forceinline__ ResultItem<T> xor_swap(ResultItem<T> x, int mask, int dir)
{
  ResultItem<T> y;
  y.dist() = __shfl_xor_sync(raft::warp_full_mask(), x.dist(), mask, raft::warp_size());
  y.id_with_flag() =
    __shfl_xor_sync(raft::warp_full_mask(), x.id_with_flag(), mask, raft::warp_size());
  return x < y == dir ? y : x;
}

__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
  int y = __shfl_xor_sync(raft::warp_full_mask(), x, mask, raft::warp_size());
  return x < y == dir ? y : x;
}

// TODO: Move to RAFT utils https://github.com/rapidsai/raft/issues/1827
__device__ __forceinline__ uint bfe(uint lane_id, uint pos)
{
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T* element_ptr, const int lane_id)
{
  static_assert(raft::warp_size() == 32);
  auto& element = *element_ptr;
  element       = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x10, bfe(lane_id, 4));
  element       = xor_swap(element, 0x08, bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 0));
  return;
}

constexpr int TILE_ROW_WIDTH = 64;
constexpr int TILE_COL_WIDTH = 32;

constexpr int NUM_SAMPLES = 32;
// For now, the max. number of samples is 32, so the sample cache size is fixed
// to 64 (32 * 2).
constexpr int MAX_NUM_BI_SAMPLES = 64;
// By adding a small padding (skew_dim), ensure that each row starts on a different memory bank, so
// warps can access memory without conflicts.
constexpr int SKEWED_MAX_NUM_BI_SAMPLES = skew_dim<float>(MAX_NUM_BI_SAMPLES);  // becomes 68
constexpr int BLOCK_SIZE                = 512;
constexpr int WMMA_M                    = 16;
constexpr int WMMA_N                    = 16;
constexpr int WMMA_K                    = 16;

template <typename Data_t>
__device__ __forceinline__ void load_vec(Data_t* vec_buffer,
                                         const Data_t* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  if constexpr (std::is_same_v<Data_t, float> or std::is_same_v<Data_t, uint8_t> or
                std::is_same_v<Data_t, int8_t>) {
    constexpr int num_load_elems_per_warp = raft::warp_size();
    for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
      int idx = step * num_load_elems_per_warp + lane_id;
      if (idx < load_dims) {
        vec_buffer[idx] = d_vec[idx];
      } else if (idx < padding_dims) {
        vec_buffer[idx] = 0.0f;
      }
    }
  }
  if constexpr (std::is_same_v<Data_t, __half>) {
    if ((size_t)d_vec % sizeof(float2) == 0 && (size_t)vec_buffer % sizeof(float2) == 0 &&
        load_dims % 4 == 0 && padding_dims % 4 == 0) {
      constexpr int num_load_elems_per_warp = raft::warp_size() * 4;
#pragma unroll
      for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx_in_vec = step * num_load_elems_per_warp + lane_id * 4;
        if (idx_in_vec + 4 <= load_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = *(float2*)(d_vec + idx_in_vec);
        } else if (idx_in_vec + 4 <= padding_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = float2({0.0f, 0.0f});
        }
      }
    } else {
      constexpr int num_load_elems_per_warp = raft::warp_size();
      for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx = step * num_load_elems_per_warp + lane_id;
        if (idx < load_dims) {
          vec_buffer[idx] = d_vec[idx];
        } else if (idx < padding_dims) {
          vec_buffer[idx] = 0.0f;
        }
      }
    }
  }
}

// TODO: Replace with RAFT utilities https://github.com/rapidsai/raft/issues/1827
/** Calculate L2 norm, and cast data to __half */
template <typename Data_t>
RAFT_KERNEL preprocess_data_kernel(
  const Data_t* input_data,
  float* output_data,
  int dim,
  DistData_t* l2_norms,
  size_t list_offset                  = 0,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
{
  extern __shared__ char buffer[];
  __shared__ float l2_norm;
  Data_t* s_vec  = (Data_t*)buffer;
  size_t list_id = list_offset + blockIdx.x;

  load_vec(s_vec, input_data + blockIdx.x * dim, dim, dim, threadIdx.x % raft::warp_size());
  if (threadIdx.x == 0) { l2_norm = 0; }
  __syncthreads();

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::CosineExpanded) {
    int lane_id = threadIdx.x % raft::warp_size();
    for (int step = 0; step < raft::ceildiv(dim, raft::warp_size()); step++) {
      int idx         = step * raft::warp_size() + lane_id;
      float part_dist = 0;
      if (idx < dim) {
        part_dist = s_vec[idx];
        part_dist = part_dist * part_dist;
      }
      __syncwarp();
      for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
        part_dist += __shfl_down_sync(raft::warp_full_mask(), part_dist, offset);
      }
      if (lane_id == 0) { l2_norm += part_dist; }
      __syncwarp();
    }
  }

  for (int step = 0; step < raft::ceildiv(dim, raft::warp_size()); step++) {
    int idx = step * raft::warp_size() + threadIdx.x;
    if (idx < dim) {
      if (metric == cuvs::distance::DistanceType::InnerProduct) {
        output_data[list_id * dim + idx] = input_data[(size_t)blockIdx.x * dim + idx];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        output_data[list_id * dim + idx] =
          (float)input_data[(size_t)blockIdx.x * dim + idx] / sqrt(l2_norm);
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        int idx_for_byte           = list_id * dim + idx;  // uint8 or int8 data
        uint8_t* output_bytes      = reinterpret_cast<uint8_t*>(output_data);
        output_bytes[idx_for_byte] = input_data[(size_t)blockIdx.x * dim + idx];
      } else {  // L2Expanded or L2SqrtExpanded
        output_data[list_id * dim + idx] = input_data[(size_t)blockIdx.x * dim + idx];
        if (idx == 0) { l2_norms[list_id] = l2_norm; }
      }
    }
  }
}

// sus: we are only taking the first few
template <typename Index_t>
RAFT_KERNEL add_rev_edges_kernel(const Index_t* graph,
                                 Index_t* rev_graph,
                                 int num_samples,
                                 int2* list_sizes)
{
  size_t list_id = blockIdx.x;
  int2 list_size = list_sizes[list_id];

  for (int idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
    // each node has same number (num_samples) of forward and reverse edges
    size_t rev_list_id = graph[list_id * num_samples + idx];
    // there are already num_samples forward edges
    int idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);
    if (idx_in_rev_list >= num_samples) {
      atomicExch(&list_sizes[rev_list_id].y, num_samples);
    } else {
      rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
    }
  }
}

template <typename Index_t, typename ID_t = InternalID_t<Index_t>>
__device__ void insert_to_global_graph(ResultItem<Index_t> elem,
                                       size_t list_id,
                                       ID_t* graph,
                                       DistData_t* dists,
                                       int node_degree,
                                       int* locks)
{
  int tx                 = threadIdx.x;
  int lane_id            = tx % raft::warp_size();
  size_t global_idx_base = list_id * node_degree;
  if (elem.id() == list_id) return;

  const int num_segments = raft::ceildiv(node_degree, raft::warp_size());

  int loop_flag = 0;
  do {
    int segment_id = elem.id() % num_segments;
    if (lane_id == 0) {
      loop_flag = atomicCAS(&locks[list_id * num_segments + segment_id], 0, 1) == 0;
    }

    loop_flag = __shfl_sync(raft::warp_full_mask(), loop_flag, 0);

    if (loop_flag == 1) {
      ResultItem<Index_t> knn_list_frag;
      int local_idx     = segment_id * raft::warp_size() + lane_id;
      size_t global_idx = global_idx_base + local_idx;
      if (local_idx < node_degree) {
        knn_list_frag.id_with_flag() = graph[global_idx].id_with_flag();
        knn_list_frag.dist()         = dists[global_idx];
      }

      int pos_to_insert = -1;
      ResultItem<Index_t> prev_elem;

      prev_elem.id_with_flag() =
        __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.id_with_flag(), 1);
      prev_elem.dist() = __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.dist(), 1);

      if (lane_id == 0) {
        prev_elem = ResultItem<Index_t>{std::numeric_limits<Index_t>::min(),
                                        std::numeric_limits<DistData_t>::lowest()};
      }
      if (elem > prev_elem && elem < knn_list_frag) {
        pos_to_insert = segment_id * raft::warp_size() + lane_id;
      } else if (elem == prev_elem || elem == knn_list_frag) {
        pos_to_insert = -2;
      }
      uint mask = __ballot_sync(raft::warp_full_mask(), pos_to_insert >= 0);
      if (mask) {
        uint set_lane_id = __fns(mask, 0, 1);
        pos_to_insert    = __shfl_sync(raft::warp_full_mask(), pos_to_insert, set_lane_id);
      }

      if (pos_to_insert >= 0) {
        int local_idx = segment_id * raft::warp_size() + lane_id;
        if (local_idx > pos_to_insert) {
          local_idx++;
        } else if (local_idx == pos_to_insert) {
          graph[global_idx_base + local_idx].id_with_flag() = elem.id_with_flag();
          dists[global_idx_base + local_idx]                = elem.dist();
          local_idx++;
        }
        size_t global_pos = global_idx_base + local_idx;
        if (local_idx < (segment_id + 1) * raft::warp_size() && local_idx < node_degree) {
          graph[global_pos].id_with_flag() = knn_list_frag.id_with_flag();
          dists[global_pos]                = knn_list_frag.dist();
        }
      }
      __threadfence();
      if (loop_flag && lane_id == 0) { atomicExch(&locks[list_id * num_segments + segment_id], 0); }
    }
  } while (!loop_flag);
}

template <typename Index_t>
__device__ ResultItem<Index_t> get_min_item(
  const Index_t id,             // the “current” row/vertex
  const int idx_in_list,        // row index in distances
  const Index_t* neighbs,       // neighbor indices for this row
  const DistData_t* distances,  // flattened distance matrix
  const bool find_in_row = true)
{
  int lane_id = threadIdx.x % raft::warp_size();

  static_assert(MAX_NUM_BI_SAMPLES == 64);
  int idx[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  float dist[MAX_NUM_BI_SAMPLES / raft::warp_size()] = {std::numeric_limits<DistData_t>::max(),
                                                        std::numeric_limits<DistData_t>::max()};
  idx[0]                                             = lane_id;
  idx[1]                                             = raft::warp_size() + lane_id;

  if (neighbs[idx[0]] != id) {
    dist[0] = find_in_row ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + lane_id]
                          : distances[idx_in_list + lane_id * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (neighbs[idx[1]] != id) {
    dist[1] =
      find_in_row
        ? distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + raft::warp_size() + lane_id]
        : distances[idx_in_list + (raft::warp_size() + lane_id) * SKEWED_MAX_NUM_BI_SAMPLES];
  }

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0]  = idx[1];
  }
  __syncwarp();
  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    float other_idx  = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist = __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0]  = other_idx;
    }
  }

  ResultItem<Index_t> result;
  result.dist()         = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  result.id_with_flag() = neighbs[__shfl_sync(raft::warp_full_mask(), idx[0], 0)];
  return result;
}

// merge two candidate lists (list_a and list_b) while removing duplicates
template <typename T>
__device__ __forceinline__ void remove_duplicates(
  T* list_a, int list_a_size, T* list_b, int list_b_size, int& unique_counter, int execute_warp_id)
{
  static_assert(raft::warp_size() == 32);
  if (!(threadIdx.x >= execute_warp_id * raft::warp_size() &&
        threadIdx.x < execute_warp_id * raft::warp_size() + raft::warp_size())) {
    return;
  }
  // okay so if over size, we put max value into it
  int lane_id = threadIdx.x % raft::warp_size();
  T elem      = std::numeric_limits<T>::max();
  if (lane_id < list_a_size) { elem = list_a[lane_id]; }
  warp_bitonic_sort(&elem, lane_id);

  if (elem != std::numeric_limits<T>::max()) { list_a[lane_id] = elem; }

  T elem_b = std::numeric_limits<T>::max();

  if (lane_id < list_b_size) { elem_b = list_b[lane_id]; }
  __syncwarp();

  int idx_l    = 0;
  int idx_r    = list_a_size;
  bool existed = false;
  while (idx_l < idx_r) {
    int idx  = (idx_l + idx_r) / 2;
    int elem = list_a[idx];
    if (elem == elem_b) {
      existed = true;
      break;
    }
    if (elem_b > elem) {
      idx_l = idx + 1;
    } else {
      idx_r = idx;
    }
  }
  if (!existed && elem_b != std::numeric_limits<T>::max()) {
    int idx                   = atomicAdd(&unique_counter, 1);
    list_a[list_a_size + idx] = elem_b;
  }
}

// launch_bounds here denote BLOCK_SIZE = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = BLOCK_SIZE * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore

// Compute pairwise distances between new-new and new-old neighbors (using L2, cosine, inner
// product, or Hamming). Update the global KNN graph with neighbors that have smaller distances,
// maintaining DEGREE_ON_DEVICE top neighbors.
template <typename Index_t, typename ID_t = InternalID_t<Index_t>, typename DistEpilogue_t>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(BLOCK_SIZE, 4)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
#endif
  local_join_kernel(const Index_t* graph_new,      // new neighbors [nrow x num_sample]
                    const Index_t* rev_graph_new,  // rev new neighbors [nrow x num_sample]
                    const int2* sizes_new,         // int2 [nrow]. x for fwd, y for bwd
                    const Index_t* graph_old,      // old neighbors [nrow x num_sample]
                    const Index_t* rev_graph_old,  // rev old neighbors [nrow x num_sample]
                    const int2* sizes_old,         // int2 [nrow]. x for fwd, y for bwd
                    const int width,
                    const float* data,
                    const int data_dim,
                    ID_t* graph,        // output graph [nrow x DEGREE_ON_DEVICE]
                    DistData_t* dists,  // output dists [nrow x DEGREE_ON_DEVICE]
                    int graph_width,
                    int* locks,
                    DistData_t* l2_norms,
                    cuvs::distance::DistanceType metric,
                    DistEpilogue_t dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];  // 64 * 2. NUM_SAMPLES is 32

  // constexpr int APAD = 8;
  // constexpr int BPAD = 8;
  // s_nv holds new neighbors’ vectors, each row is a vector of TILE_COL_WIDTH (128) elements (plus
  // padding (8) for alignment)
  __shared__ float s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH];                     // New vectors
  __shared__ float s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH];                     // Old vectors
  __shared__ float s_distances[MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES];  // Old vectors
  // static_assert(sizeof(float) * MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES <=
  //               sizeof(float) * MAX_NUM_BI_SAMPLES * (TILE_COL_WIDTH + BPAD));
  // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov

  // 4 * MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES == 2 * MAX_NUM_BI_SAMPLES * (TILE_COL_WIDTH
  // + APAD)
  // float* s_distances =
  //   (float*)&s_ov[0][0];  // perfectly works for MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES
  int* s_unique_counter = (int*)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  // The kernel combines neighbors and reverse neighbors into shared memory arrays.
  // These are all the potential neighbors the kernel will evaluate for node i in this iteration.

  // sus: for cases where list_new_size2.x is not the full NUM_SAMPLES(32) what happens to the empty
  // slots? what is graph_new init with?
  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  // if we have a distance epilogue, distances need to be fully calculated instead of postprocessing
  // them.
  bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;

  // Deduplicate the neighbor lists before computing distances.
  // After this step: list_new_size and list_old_size are updated to the number of unique neighbors.
  // removing within the "new" OR "old" graph to make sure we don't have duplicates between the
  // reverse graph and the base graph
  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  // now we have a unified list in new_neighbors, and old_neighbors
  // each of size list_new_size and list_old_size respectively
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  // block size is 512, so there are 16 warps.
  // 32 threads per warp
  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();  // 16

  // This is a 2D tiling of warps: 16 warps are arranged as a 4×4 grid
  // int warp_id_y = warp_id / 4;
  // int warp_id_x = warp_id % 4;

  // CUDA Tensor Core fragments: small pieces of matrices loaded into registers for fast
  // multiply-accumulate. a_frag is row-major, b_frag is column-major. c_frag accumulates the result
  // in FP32.
  // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> a_frag;
  // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::col_major> b_frag;
  // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  //   if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
  //     wmma::fill_fragment(c_frag, 0.0);  // initialize accumulator to 0

  //     // The vectors are split into tiles of width TILE_COL_WIDTH to fit in shared memory.
  //     for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
  //       int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
  //                              ? data_dim - step * TILE_COL_WIDTH
  //                              : TILE_COL_WIDTH;
  //       // Each warp loads multiple new neighbors’ vectors from global memory into shared memory
  //       s_nv
  //       // in a tiled and lane-parallel fashion. load_vec ensures that each thread in a warp
  //       loads
  //       // part of a vector.
  // #pragma unroll
  //       for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
  //         int idx = i * num_warps + warp_id;
  //         if (idx < list_new_size) {
  //           size_t neighbor_id = new_neighbors[idx];
  //           size_t idx_in_data = neighbor_id * data_dim;
  //           load_vec(s_nv[idx],
  //                    data + idx_in_data + step * TILE_COL_WIDTH,
  //                    num_load_elems,
  //                    TILE_COL_WIDTH,
  //                    lane_id);
  //         }
  //       }
  //       __syncthreads();

  //       for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
  //         wmma::load_matrix_sync(
  //           a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
  //         wmma::load_matrix_sync(
  //           b_frag, s_nv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
  //         wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  //         __syncthreads();
  //       }
  //     }

  //     wmma::store_matrix_sync(
  //       s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
  //       c_frag,
  //       SKEWED_MAX_NUM_BI_SAMPLES,
  //       wmma::mem_row_major);
  //   }

  // Zero out distance matrix
  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    int tid = threadIdx.x;
    for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x)
      s_distances[i] = 0.0f;

    __syncthreads();

    // Loop over tiles of the vector dimension
    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
      // Each warp loads multiple new neighbors’ vectors from global memory into shared memory s_nv
      // in a tiled and lane-parallel fashion. load_vec ensures that each thread in a warp loads
      // part of a vector.
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }  // loaded vectors
      __syncthreads();

      if (tid == 0) {
        for (int row = 0; row < list_new_size; row++) {
          for (int col = 0; col < list_new_size; col++) {
            float acc = 0.0f;
            for (int d = 0; d < num_load_elems; d++) {
              acc += s_nv[row][d] * s_nv[col][d];
            }
            s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col] += acc;
          }
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();

  // s_distances in this kernel is essentially acting as a small pairwise distance (or similarity)
  // matrix between the "sample" vectors MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES = total
  // number of "cells" in a virtual 2D distance matrix we want to compute.
  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    int row_id = i % SKEWED_MAX_NUM_BI_SAMPLES;
    int col_id = i / SKEWED_MAX_NUM_BI_SAMPLES;

    // following the python implementation of comparing only with later new neighbors
    if (row_id < list_new_size && col_id < list_new_size && row_id <= col_id) {
      if (metric == cuvs::distance::DistanceType::InnerProduct && can_postprocess_dist) {
        s_distances[i] = -s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        s_distances[i] = 1.0 - s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        s_distances[i] = 0.0;
        int n1         = new_neighbors[row_id];
        int n2         = new_neighbors[col_id];
        // TODO: https://github.com/rapidsai/cuvs/issues/1127
        const uint8_t* data_n1 = reinterpret_cast<const uint8_t*>(data) + n1 * data_dim;
        const uint8_t* data_n2 = reinterpret_cast<const uint8_t*>(data) + n2 * data_dim;
        for (int d = 0; d < data_dim; d++) {
          s_distances[i] += __popc(static_cast<uint32_t>(data_n1[d] ^ data_n2[d]) & 0xff);
        }
      } else {  // L2Expanded or L2SqrtExpanded
        s_distances[i] =
          l2_norms[new_neighbors[row_id]] + l2_norms[new_neighbors[col_id]] - 2.0 * s_distances[i];
        // for fp32 vs fp16 precision differences resulting in negative distances when distance
        // should be 0 related issue: https://github.com/rapidsai/cuvs/issues/991
        s_distances[i] = s_distances[i] < 0.0f ? 0.0f : s_distances[i];
        if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          s_distances[i] = sqrtf(s_distances[i]);
        }
      }
      s_distances[i] = dist_epilogue(s_distances[i], new_neighbors[row_id], new_neighbors[col_id]);
      if ((new_neighbors[row_id] == 0 && new_neighbors[col_id] == 1006) ||
          (new_neighbors[row_id] == 0 && new_neighbors[col_id] == 1006)) {
        printf(
          "in row %d new-new (0, 1006) dist %f\n", static_cast<int>(blockIdx.x), s_distances[i]);
      }
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }

  __syncthreads();
  // at this point s_distances contains the final distances ready for selection of nearest
  // neighbors.

  // list_new_size is the number of new neighbors we want to process.
  // num_warps is the number of warps per block (16 in your case).
  // Each warp is responsible for one candidate at a time.
  // SUS: in original impl we try adding all potential neighbors into the list.
  // here we pick min new-new and push that to only the idx_in_list's list
  // for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
  //   // all threads in same warp end up with same idx_in_list
  //   // idx_in_list is the index of the current neighbor this warp is responsible for.
  //   int idx_in_list =
  //     step * num_warps +
  //     tx / raft::warp_size();  // tx / raft::warp_size() gives which warp this thread belongs to.
  //   if (idx_in_list >= list_new_size) continue;
  //   auto min_elem =
  //     get_min_item(new_neighbors[idx_in_list], idx_in_list, new_neighbors, s_distances);
  //   if (min_elem.id() < gridDim.x) {
  //     insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
  //   }
  // }
  // sus: do we not need to init grph and dists every time before we call local_join_kernel>
  // or update it with the altest updates of host side h_dists and h_graph?

  // This version matches the python implementation
  // Each warp processes one "list_id" at a time (idx_in_list)
  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int warp_id = tx / raft::warp_size();  // current warp id
    int idx_in_list =
      step * num_warps + warp_id;  // index of the new neighbor this warp is responsible for
    // e.g. threads in warp0 will look at neighbor 0->16->32 ...
    if (idx_in_list >= list_new_size) continue;

    // Loop through whole thing instead of just the upper triangle to avoid cross-warp race
    // conditions
    for (int candidate_idx = 0; candidate_idx < list_new_size; candidate_idx++) {
      float dist_btw_new_new = s_distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + candidate_idx];
      // don't care about segments
      // sus: okay so now this should work with only 1 segment at least
      insert_to_global_graph(ResultItem<Index_t>(new_neighbors[candidate_idx], dist_btw_new_new),
                             new_neighbors[idx_in_list],
                             graph,
                             dists,
                             graph_width,
                             locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();
  // if (threadIdx.x == 0) {
  //   printf("row %d has old values: %d\n", static_cast<int>(blockIdx.x),
  //   static_cast<int>(list_old_size));
  // }

  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    // wmma::fill_fragment(c_frag, 0.0);
    int tid = threadIdx.x;
    for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x)
      s_distances[i] = 0.0f;

    __syncthreads();

    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
      // if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      // }
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_old_size) {
          size_t neighbor_id = old_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_ov[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
          // __syncthreads();
          // if (static_cast<int>(neighbor_id) == 0 && static_cast<int>(warp_id) == 0) {
          //   printf("in row %d (0) vec [%f, %f, %f, %f]\n", static_cast<int>(blockIdx.x),
          //         static_cast<float>(s_ov[idx][0]), static_cast<float>(s_ov[idx][1]),
          //         static_cast<float>(s_ov[idx][2]), static_cast<float>(s_ov[idx][3]));
          // }
          // if (static_cast<int>(neighbor_id) == 61 && static_cast<int>(warp_id) == 0) {
          //   printf("in row %d (61) vec [%f, %f, %f, %f]\n", static_cast<int>(blockIdx.x),
          //         static_cast<float>(s_ov[idx][0]), static_cast<float>(s_ov[idx][1]),
          //         static_cast<float>(s_ov[idx][2]), static_cast<float>(s_ov[idx][3]));
          // }
          // looks like these are being loaded properly
        }
      }
      __syncthreads();

      // for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
      //   wmma::load_matrix_sync(
      //     a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
      //   wmma::load_matrix_sync(
      //     b_frag, s_ov[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
      //   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      //   __syncthreads();
      // }

      if (tid == 0) {
        for (int row = 0; row < list_new_size; row++) {
          for (int col = 0; col < list_old_size; col++) {
            float acc = 0.0f;
            for (int d = 0; d < num_load_elems; d++) {
              acc += s_nv[row][d] * s_ov[col][d];
            }
            s_distances[row * SKEWED_MAX_NUM_BI_SAMPLES + col] += acc;
          }
        }
      }
      __syncthreads();
    }
  }

  //   wmma::store_matrix_sync(
  //     s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
  //     c_frag,
  //     SKEWED_MAX_NUM_BI_SAMPLES,
  //     wmma::mem_row_major);
  //   __syncthreads();
  // }

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    int col_id = i % SKEWED_MAX_NUM_BI_SAMPLES;  // old
    int row_id = i / SKEWED_MAX_NUM_BI_SAMPLES;  // new
    if (row_id < list_new_size && col_id < list_old_size) {
      // if ((new_neighbors[row_id] == 2 && old_neighbors[col_id] == 4278) || (new_neighbors[row_id]
      // == 4278 && old_neighbors[col_id] == 2)) {
      //   printf("in row %d (2,4278) dist %f l2 norm [%f, %f]\n", static_cast<int>(blockIdx.x),
      //   s_distances[i], static_cast<float>(l2_norms[new_neighbors[row_id]]),
      //   static_cast<float>(l2_norms[old_neighbors[col_id]]));
      // }
      // if ((new_neighbors[row_id] == 2 && old_neighbors[col_id] == 7789) || (new_neighbors[row_id]
      // == 7789 && old_neighbors[col_id] == 2)) {
      //   printf("in row %d (2,7789) dist %f l2 norm [%f, %f]\n", static_cast<int>(blockIdx.x),
      //   s_distances[i], static_cast<float>(l2_norms[new_neighbors[row_id]]),
      //   static_cast<float>(l2_norms[old_neighbors[col_id]]));
      // }
      if ((new_neighbors[row_id] == 2 && old_neighbors[col_id] == 4860) ||
          (new_neighbors[row_id] == 4860 && old_neighbors[col_id] == 2)) {
        printf("in row %d (2,4860) dist %f l2 norm [%f, %f]\n",
               static_cast<int>(blockIdx.x),
               s_distances[i],
               static_cast<float>(l2_norms[new_neighbors[row_id]]),
               static_cast<float>(l2_norms[old_neighbors[col_id]]));
      }

      if (metric == cuvs::distance::DistanceType::InnerProduct && can_postprocess_dist) {
        s_distances[i] = -s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        s_distances[i] = 1.0 - s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        s_distances[i] = 0.0;
        int n1         = new_neighbors[row_id];
        int n2         = old_neighbors[col_id];
        // TODO: https://github.com/rapidsai/cuvs/issues/1127
        const uint8_t* data_n1 = reinterpret_cast<const uint8_t*>(data) + n1 * data_dim;
        const uint8_t* data_n2 = reinterpret_cast<const uint8_t*>(data) + n2 * data_dim;
        for (int d = 0; d < data_dim; d++) {
          s_distances[i] += __popc(static_cast<uint32_t>(data_n1[d] ^ data_n2[d]) & 0xff);
        }
      } else {  // L2Expanded or L2SqrtExpanded
        s_distances[i] =
          l2_norms[new_neighbors[row_id]] + l2_norms[old_neighbors[col_id]] - 2.0 * s_distances[i];
        // for fp32 vs fp16 precision differences resulting in negative distances when distance
        // should be 0 related issue: https://github.com/rapidsai/cuvs/issues/991
        s_distances[i] = s_distances[i] < 0.0f ? 0.0f : s_distances[i];
        if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          s_distances[i] = sqrtf(s_distances[i]);
        }
      }
      s_distances[i] = dist_epilogue(s_distances[i], old_neighbors[row_id], new_neighbors[col_id]);
      if ((new_neighbors[row_id] == 0 && old_neighbors[col_id] == 1006) ||
          (new_neighbors[row_id] == 0 && old_neighbors[col_id] == 1006)) {
        printf(
          "in row %d old-new (0, 1006) dist %f\n", static_cast<int>(blockIdx.x), s_distances[i]);
      }
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }
  __syncthreads();

  // for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
  //   int idx_in_list = step * num_warps + tx / raft::warp_size();
  //   if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
  //   if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES *
  //   2)
  //     continue;
  //   ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
  //                                std::numeric_limits<DistData_t>::max()};
  //   if (idx_in_list < MAX_NUM_BI_SAMPLES) {
  //     auto temp_min_item =
  //       get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
  //     if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
  //   } else {
  //     auto temp_min_item = get_min_item(
  //       s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances,
  //       false);
  //     if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
  //   }

  //   if (min_elem.id() < gridDim.x) {
  //     insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
  //   }
  // }

  // introducing this doesn't add that much improvement in recall
  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();

    // skip unused slots in the combined new+old neighbor arrays.
    if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
      continue;

    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      for (int candidate_idx = 0; candidate_idx < list_old_size; candidate_idx++) {
        float dist_btw_new_old =
          s_distances[idx_in_list * SKEWED_MAX_NUM_BI_SAMPLES + candidate_idx];
        insert_to_global_graph(ResultItem<Index_t>(old_neighbors[candidate_idx], dist_btw_new_old),
                               new_neighbors[idx_in_list],
                               graph,
                               dists,
                               graph_width,
                               locks);
      }

    } else {
      idx_in_list = idx_in_list - MAX_NUM_BI_SAMPLES;
      for (int candidate_idx = 0; candidate_idx < list_new_size; candidate_idx++) {
        float dist_btw_old_new =
          s_distances[candidate_idx * SKEWED_MAX_NUM_BI_SAMPLES + idx_in_list];
        insert_to_global_graph(ResultItem<Index_t>(new_neighbors[candidate_idx], dist_btw_old_new),
                               old_neighbors[idx_in_list],
                               graph,
                               dists,
                               graph_width,
                               locks);
      }
    }
  }
#endif
}

namespace {
// keeps each segment in sorted order by distance
// that's why we ened the width here
template <typename Index_t>
int insert_to_ordered_list(InternalID_t<Index_t>* list,
                           DistData_t* dist_list,
                           const int width,
                           const InternalID_t<Index_t> neighb_id,
                           const DistData_t dist)
{
  if (dist > dist_list[width - 1]) { return width; }

  int idx_insert      = width;
  bool position_found = false;
  for (int i = 0; i < width; i++) {
    if (list[i].id() == neighb_id.id()) { return width; }
    if (!position_found && dist_list[i] > dist) {
      idx_insert     = i;
      position_found = true;
    }
  }
  if (idx_insert == width) return idx_insert;

  memmove(list + idx_insert + 1, list + idx_insert, sizeof(*list) * (width - idx_insert - 1));
  memmove(dist_list + idx_insert + 1,
          dist_list + idx_insert,
          sizeof(*dist_list) * (width - idx_insert - 1));

  list[idx_insert]      = neighb_id;
  dist_list[idx_insert] = dist;
  return idx_insert;
};

}  // namespace

template <typename Index_t>
GnndGraph<Index_t>::GnndGraph(raft::resources const& res,
                              const size_t nrow,
                              const size_t node_degree,
                              const size_t internal_node_degree,
                              const size_t num_samples)
  : res(res),
    nrow(nrow),
    node_degree(node_degree),
    num_samples(num_samples),
    bloom_filter(nrow, internal_node_degree / segment_size, 3),
    // These are the per-iteration CPU-side staging areas for candidate neighbors:
    // Host (CPU) copy of the neighbor distances for each node.
    h_dists{raft::make_host_matrix<DistData_t, size_t, raft::row_major>(nrow, node_degree)},
    // For each node, this holds the newly sampled candidate neighbors in the current iteration.
    // Produced by sample_graph() (first iteration) or sample_graph_new() (subsequent iterations).
    // Gets copied to device (d_list_sizes_new_, h_rev_graph_new_) so GPU kernels can process them.
    h_graph_new{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    // lets GPU know how many neighbors in h_graph_new[i] are actually valid.
    h_list_sizes_new{raft::make_pinned_vector<int2, size_t>(res, nrow)},
    // For each node, this stores old neighbors (carried over from previous iterations).
    // sample_graph(false) fills this from the existing graph.
    h_graph_old{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    // counts how many valid old neighbors exist.
    h_list_sizes_old{raft::make_pinned_vector<int2, size_t>(res, nrow)}
{
  // node_degree must be a multiple of segment_size;
  assert(node_degree % segment_size == 0);
  assert(internal_node_degree % segment_size == 0);

  num_segments = node_degree / segment_size;
  // To save the CPU memory, graph should be allocated by external function
  h_graph = nullptr;
  // std::cout << "node degree " << node_degree << " internal node degree " << internal_node_degree
  //           << " num samples " << num_samples << " num_segments " << num_segments << std::endl;
}

// This is the only operation on the CPU that cannot be overlapped.
// So it should be as fast as possible.
// Input source: a flat array of candidate neighbors (new_neighbors) just produced by some batch of
// GPU kernels (e.g. neighbors-of-neighbors comparisons). cleans and deduplicates GPU-generated
// candidates before the next round.
template <typename Index_t>
void GnndGraph<Index_t>::sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    auto list_new                       = h_graph_new.data_handle() + i * num_samples;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighb_id >= nrow) break;
      if (bloom_filter.check(i, new_neighb_id)) { continue; }
      bloom_filter.add(i, new_neighb_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new.data_handle()[i].x++] = new_neighb_id;
      if (h_list_sizes_new.data_handle()[i].x == num_samples) break;
    }
  }
}

// initializing to add diversity to the init
// functioning as the RP tree init in pynndescent
// segments: partition dataset into disjoint slices so that each node's neighbors list is
// initialized in balanced chunks neighbors look like this: [ seg_0 | seg_1 | ... |
// seg_{num_segments-1} ]

template <typename Index_t>
void GnndGraph<Index_t>::init_random_graph()
{
  std::cout << "initializing random graph. number of segments is " << num_segments << std::endl;
  for (size_t seg_idx = 0; seg_idx < static_cast<size_t>(num_segments); seg_idx++) {
    // random sequence (range: 0~nrow)
    // segment_x stores neighbors which id % num_segments == x
    // rand gen of sequential integers in segment size
    std::vector<Index_t> rand_seq((nrow + num_segments - 1) / num_segments);
    std::iota(rand_seq.begin(), rand_seq.end(), 0);
    auto gen = std::default_random_engine{seg_idx};
    std::shuffle(rand_seq.begin(), rand_seq.end(), gen);  // fixed random permutation for

#pragma omp parallel for
    for (size_t i = 0; i < nrow; i++) {
      size_t base_idx         = i * node_degree + seg_idx * segment_size;
      auto h_neighbor_list    = h_graph + base_idx;
      auto h_dist_list        = h_dists.data_handle() + base_idx;
      size_t idx              = base_idx;
      size_t self_in_this_seg = 0;
      for (size_t j = 0; j < static_cast<size_t>(segment_size); j++) {
        // each segment only uses IDs with id % num_segments == seg_idx
        // reconstructs the actual global ID belonging to that segment.
        // each segment covers a different residue class modulo num_segments → so the random
        // neighbors are spread evenly across the dataset.
        Index_t id = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;  // some random value
        if ((size_t)id == i) {
          idx++;
          id               = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
          self_in_this_seg = 1;
        }

        // not sure why we need first half of this condition
        h_neighbor_list[j].id_with_flag() =
          j < (rand_seq.size() - self_in_this_seg) && size_t(id) < nrow
            ? id
            : std::numeric_limits<Index_t>::max();
        h_dist_list[j] = std::numeric_limits<DistData_t>::max();
        idx++;
      }
    }
  }
}

// Input source: the full segmented neighbor list (h_graph) for each node.
// Purpose: sample a mixture of old and new neighbors (depending on the sample_new flag) from across
// all segments of the current graph. Traverses neighbor lists in a row-by-row, segment-by-segment
// pattern to ensure coverage across segments. Marks new neighbors as “old” once sampled, so they
// won’t be re-sampled infinitely.
// Stores results into h_graph_old and h_graph_new.
// Role in algorithm: this is the “normal” per-iteration sampling step that drives NN-Descent’s
// refinement (neighbors-of-neighbors search).
template <typename Index_t>
void GnndGraph<Index_t>::sample_graph(bool sample_new)
{
  // // #pragma omp parallel for
  //   auto shuffled_list = raft::make_host_vector<InternalID_t<Index_t>>(node_degree);

  //   std::random_device rd;
  //   std::mt19937 gen(rd());

  // sus: maybe random shuffle here instead of doing this in orfer evey time

  // fill in with max sentinel value
  std::fill_n(h_graph_old.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
  std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());

  for (size_t i = 0; i < nrow; i++) {
    h_list_sizes_old.data_handle()[i].x = 0;
    h_list_sizes_old.data_handle()[i].y = 0;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list     = h_graph + i * node_degree;
    auto list_old = h_graph_old.data_handle() + i * num_samples;
    auto list_new = h_graph_new.data_handle() + i * num_samples;

    // raft::copy(shuffled_list.data_handle(), list, node_degree,
    // raft::resource::get_cuda_stream(res));
    // // shuffle by segment
    // for (int k = 0; k < num_segments; k++) {
    //     auto seg_start = shuffled_list.data_handle() + k * segment_size;
    //     auto seg_end   = seg_start + segment_size;
    //     std::shuffle(seg_start, seg_end, gen);
    // }

    // Round 1: pick 1 neighbor from each segment
    // Round 2: pick the next neighbor from each segment
    // gives balanced exploration of old/new neighbors across the graph, instead of sampling many
    // from the same local “region” sus: maybe worth random shuffling per segment hgraph instead of
    // doing in-order?
    for (int j = 0; j < segment_size; j++) {
      for (int k = 0; k < num_segments; k++) {
        auto neighbor = list[k * segment_size + j];
        if ((size_t)neighbor.id() >= nrow) continue;
        if (!neighbor.is_new()) {  // old neighbor
          if (h_list_sizes_old.data_handle()[i].x < num_samples) {
            list_old[h_list_sizes_old.data_handle()[i].x++] = neighbor.id();
          }
        } else if (sample_new) {  // neighbor is new and sample_new
          if (h_list_sizes_new.data_handle()[i].x < num_samples) {
            list[k * segment_size + j].mark_old();  // we mark this old now that we sample it
            list_new[h_list_sizes_new.data_handle()[i].x++] = neighbor.id();
          }
        }
        if (h_list_sizes_old.data_handle()[i].x == num_samples &&
            h_list_sizes_new.data_handle()[i].x == num_samples) {
          break;
        }
      }
      if (h_list_sizes_old.data_handle()[i].x == num_samples &&
          h_list_sizes_new.data_handle()[i].x == num_samples) {
        break;
      }  // stop once we have enough old AND new neighbors
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::update_graph(const InternalID_t<Index_t>* new_neighbors,
                                      const DistData_t* new_dists,
                                      const size_t width,
                                      std::atomic<int64_t>& update_counter)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j];
      auto new_dist      = new_dists[i * width + j];
      if (new_dist == std::numeric_limits<DistData_t>::max()) break;
      if ((size_t)new_neighb_id.id() == i) continue;
      int seg_idx = new_neighb_id.id() % num_segments;
      auto list   = h_graph + i * node_degree + seg_idx * segment_size;
      // neighbor x is always stored in the segment x % num_segments
      auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * segment_size;
      if (i == 0 && new_neighb_id.id() == 1006) {
        std::cout << "row 0 neighbor 1006 adding to segment " << seg_idx << std::endl;
      }
      int insert_pos =
        insert_to_ordered_list(list, dist_list, segment_size, new_neighb_id, new_dist);
      if (i % counter_interval == 0 && insert_pos != segment_size) { update_counter++; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sort_lists()
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    std::vector<std::pair<DistData_t, Index_t>> new_list;
    for (size_t j = 0; j < node_degree; j++) {
      new_list.emplace_back(h_dists.data_handle()[i * node_degree + j],
                            h_graph[i * node_degree + j].id());
    }
    std::sort(new_list.begin(), new_list.end());
    for (size_t j = 0; j < node_degree; j++) {
      h_graph[i * node_degree + j].id_with_flag() = new_list[j].second;
      h_dists.data_handle()[i * node_degree + j]  = new_list[j].first;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::clear()
{
  bloom_filter.clear();
}

template <typename Index_t>
GnndGraph<Index_t>::~GnndGraph()
{
  assert(h_graph == nullptr);
}

// global device/host workspaces for maintaining the graph and distances:
template <typename Data_t, typename Index_t>
GNND<Data_t, Index_t>::GNND(raft::resources const& res, const BuildConfig& build_config)
  : res(res),
    build_config_(build_config),
    graph_(res,
           build_config.max_dataset_size,
           align32::roundUp(build_config.node_degree),
           align32::roundUp(build_config.internal_node_degree ? build_config.internal_node_degree
                                                              : build_config.node_degree),
           NUM_SAMPLES),
    nrow_(build_config.max_dataset_size),
    ndim_(build_config.dataset_dim),
    d_data_{raft::make_device_matrix<float, size_t, raft::row_major>(
      res,
      nrow_,
      build_config.metric == cuvs::distance::DistanceType::BitwiseHamming
        ? (build_config.dataset_dim + 1) / 2
        : build_config.dataset_dim)},
    l2_norms_{raft::make_device_vector<DistData_t, size_t>(res, 0)},
    // GPU-resident neighbor list (adjacency list) — the main evolving k-NN graph.
    // Holds the best node_degree neighbors per node.
    graph_buffer_{
      raft::make_device_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    // Updated in sync with graph_buffer_.
    // Reused temporarily in add_reverse_edges() (hence the static_assert ensuring space).
    dists_buffer_{
      raft::make_device_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    // Host mirror of graph_buffer_
    // Needed because sample_graph_new() runs on CPU (OpenMP).
    // Copied from device → host after GPU kernels finish, before CPU resampling.
    graph_host_buffer_{
      raft::make_pinned_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    // Host mirror of dists_buffer_.
    // supports CPU-side operations (update_and_sample thread).
    dists_host_buffer_{
      raft::make_pinned_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    // Per-row spinlocks used on GPU when multiple threads attempt to update the same node’s
    // neighbor list concurrently.
    d_locks_{raft::make_device_vector<int, size_t>(res, nrow_)},
    // Reverse edges of h_graph_new.
    h_rev_graph_new_{
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)},
    h_graph_old_(
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)),
    // Reverse edges of h_graph_old. Same role, but for old neighbors.
    h_rev_graph_old_{
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)},
    d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, nrow_)},
    d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, nrow_)}
{
  static_assert(NUM_SAMPLES <= 32);

  std::cout << "initializing graph_buffer_ and dists_buffer_\n";

  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);

  if (build_config.metric == cuvs::distance::DistanceType::L2Expanded ||
      build_config.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    l2_norms_ = raft::make_device_vector<DistData_t, size_t>(res, nrow_);
  }
};

template <typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::reset(raft::resources const& res)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);
}

template <typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::add_reverse_edges(Index_t* graph_ptr,
                                              Index_t* h_rev_graph_ptr,
                                              Index_t* d_rev_graph_ptr,
                                              int2* list_sizes,
                                              cudaStream_t stream)
{
  raft::matrix::fill(
    res,
    raft::make_device_matrix_view<Index_t, int64_t>(d_rev_graph_ptr, nrow_, DEGREE_ON_DEVICE),
    std::numeric_limits<Index_t>::max());
  // raft::print_host_vector("\tadd rev edges input graph", graph_ptr, NUM_SAMPLES, std::cout);
  cudaDeviceSynchronize();
  // raft::print_device_vector("\tadd rev edges emp buffer", d_rev_graph_ptr, NUM_SAMPLES,
  // std::cout);
  add_rev_edges_kernel<<<nrow_, raft::warp_size(), 0, stream>>>(
    graph_ptr, d_rev_graph_ptr, NUM_SAMPLES, list_sizes);
  // cudaDeviceSynchronize();
  // raft::print_device_vector("\tadd rev edges emp buffer after", d_rev_graph_ptr, NUM_SAMPLES,
  // std::cout);
  raft::copy(h_rev_graph_ptr, d_rev_graph_ptr, nrow_ * NUM_SAMPLES, stream);
  cudaDeviceSynchronize();
  // raft::print_host_vector("\tadd rev edges output graph", h_rev_graph_ptr, NUM_SAMPLES,
  // std::cout);
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::local_join(cudaStream_t stream, DistEpilogue_t dist_epilogue)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  local_join_kernel<<<nrow_, BLOCK_SIZE, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                      h_rev_graph_new_.data_handle(),
                                                      d_list_sizes_new_.data_handle(),
                                                      graph_.h_graph_old.data_handle(),
                                                      h_rev_graph_old_.data_handle(),
                                                      d_list_sizes_old_.data_handle(),
                                                      NUM_SAMPLES,
                                                      d_data_.data_handle(),
                                                      ndim_,
                                                      graph_buffer_.data_handle(),
                                                      dists_buffer_.data_handle(),
                                                      DEGREE_ON_DEVICE,
                                                      d_locks_.data_handle(),
                                                      l2_norms_.data_handle(),
                                                      build_config_.metric,
                                                      dist_epilogue);
}

// sample_graph(true) (bootstrap only):
// Pulls both new + old neighbors from the initial random graph.

// sample_graph(false) (each iteration):
// Samples only old neighbors from the segmented graph (steady exploration).
// → Runs in parallel with GPU work.

// sample_graph_new (each iteration, after GPU kernels):
// Processes new candidate neighbors from GPU, dedupes them, and prepares them for the next round.

// Contextual Flow
// Initialization
// graph_buffer_ + dists_buffer_ start empty (max() values).
// sample_graph(true) seeds h_graph_new + h_graph_old.

// Each iteration
// Copy h_graph_new/old sizes → device (d_list_sizes_new_, etc.).
// GPU computes candidate distances (local_join).
// update_graph() updates graph_buffer_ + dists_buffer_ with better neighbors.
// CPU thread (update_and_sample) does resampling (sample_graph(false)), preparing the next
// iteration’s h_graph_old. After GPU finishes, copy updated graph_buffer_ back to host
// (graph_host_buffer_) and run sample_graph_new() to fill h_graph_new from the latest graph state.
// Repeat.

// Termination
// Controlled by update_counter_ → if too few updates happen, stop early.

// template <typename Data_t, typename Index_t>
// template <typename DistEpilogue_t>
// void GNND<Data_t, Index_t>::build(Data_t* data,
//                                   const Index_t nrow,
//                                   Index_t* output_graph,
//                                   bool return_distances,
//                                   DistData_t* output_distances,
//                                   DistEpilogue_t dist_epilogue)
// {
//   using input_t = typename std::remove_const<Data_t>::type;
//   if (build_config_.metric == cuvsDistanceType::BitwiseHamming &&
//       !(std::is_same_v<input_t, uint8_t> || std::is_same_v<input_t, int8_t>)) {
//     RAFT_FAIL(
//       "Data type needs to be int8 or uint8 for NN Descent to run with BitwiseHamming distance.");
//   }

//   cudaStream_t stream = raft::resource::get_cuda_stream(res);
//   nrow_               = nrow;
//   graph_.nrow         = nrow;
//   graph_.bloom_filter.set_nrow(nrow);
//   update_counter_ = 0;
//   graph_.h_graph  = (InternalID_t<Index_t>*)output_graph;
//   raft::matrix::fill(res, d_data_.view(), static_cast<__half>(0));

//   cudaPointerAttributes data_ptr_attr;
//   RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
//   size_t batch_size = (data_ptr_attr.devicePointer == nullptr) ? 100000 : nrow_;

//   cuvs::spatial::knn::detail::utils::batch_load_iterator vec_batches{
//     data, static_cast<size_t>(nrow_), build_config_.dataset_dim, batch_size, stream};
//   for (auto const& batch : vec_batches) {
//     preprocess_data_kernel<<<
//       batch.size(),
//       raft::warp_size(),
//       sizeof(Data_t) * ceildiv(build_config_.dataset_dim, static_cast<size_t>(raft::warp_size()))
//       *
//         raft::warp_size(),
//       stream>>>(batch.data(),
//                 d_data_.data_handle(),
//                 build_config_.dataset_dim,
//                 l2_norms_.data_handle(),
//                 batch.offset(),
//                 build_config_.metric);
//   }

//   graph_.clear(); // clearing bloom filter
//   graph_.init_random_graph(); // this just initializes the h_graph
//   // filling in h_graph_old and h_graph_new, sets up first round of neighbor comparisons
//   graph_.sample_graph(true);

//   // if true:
//   // Merge new candidates back into the segmented graph via update_graph(...).
//   // ensures the main neighbor lists remain sorted and balanced (per segment).
//   // Also increments update_counter_ to monitor convergence.
//   auto update_and_sample = [&](bool update_graph) {
//     if (update_graph) {
//       update_counter_ = 0;
//       graph_.update_graph(graph_host_buffer_.data_handle(),
//                           dists_host_buffer_.data_handle(),
//                           DEGREE_ON_DEVICE,
//                           update_counter_);
//       if (update_counter_ < build_config_.termination_threshold * nrow_ *
//                               build_config_.dataset_dim / counter_interval) {
//         update_counter_ = -1;
//       }
//     }
//     // samples only old neighbors (already in the graph) into h_graph_old.
//     // Why? Because during the same iteration, fresh candidates will come from the GPU side — no
//     need to pre-sample them again from the graph.
//     // just old neighbors in parallel with GPU work.
//     graph_.sample_graph(false);
//   };

//   for (size_t it = 0; it < build_config_.max_iterations; it++) {
//     raft::copy(d_list_sizes_new_.data_handle(),
//                graph_.h_list_sizes_new.data_handle(),
//                nrow_,
//                raft::resource::get_cuda_stream(res));
//     raft::copy(h_graph_old_.data_handle(),
//                graph_.h_graph_old.data_handle(),
//                nrow_ * NUM_SAMPLES,
//                raft::resource::get_cuda_stream(res));
//     raft::copy(d_list_sizes_old_.data_handle(),
//                graph_.h_list_sizes_old.data_handle(),
//                nrow_,
//                raft::resource::get_cuda_stream(res));
//     raft::resource::sync_stream(res);

//     // sus: why are we doing another old graph sampling here
//     std::thread update_and_sample_thread(update_and_sample, it);

//     RAFT_LOG_DEBUG("# GNND iteraton: %lu / %lu", it + 1, build_config_.max_iterations);

//     // GPU kernels add_reverse_edges and local_join:
//     // GPU compares sampled neighbors (old and new) and produces candidate new neighbors for each
//     node.
//     // These candidates are written into graph_buffer_ / dists_buffer_.

//     // Reuse dists_buffer_ to save GPU memory. graph_buffer_ cannot be reused, because it
//     // contains some information for local_join.
//     static_assert(DEGREE_ON_DEVICE * sizeof(*(dists_buffer_.data_handle())) >=
//                   NUM_SAMPLES * sizeof(*(graph_buffer_.data_handle())));
//     add_reverse_edges(graph_.h_graph_new.data_handle(),
//                       h_rev_graph_new_.data_handle(),
//                       (Index_t*)dists_buffer_.data_handle(),
//                       d_list_sizes_new_.data_handle(),
//                       stream);
//     add_reverse_edges(h_graph_old_.data_handle(),
//                       h_rev_graph_old_.data_handle(),
//                       (Index_t*)dists_buffer_.data_handle(),
//                       d_list_sizes_old_.data_handle(),
//                       stream);

//     // Tensor operations from `mma.h` are guarded with archicteture
//     // __CUDA_ARCH__ >= 700. Since RAFT supports compilation for ARCH 600,
//     // we need to ensure that `local_join_kernel` (which uses tensor) operations
//     // is not only not compiled, but also a runtime error is presented to the user
//     auto kernel       = preprocess_data_kernel<input_t>;
//     void* kernel_ptr  = reinterpret_cast<void*>(kernel);
//     auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
//     auto wmma_range =
//       raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

//     if (wmma_range.contains(runtime_arch)) {
//       local_join(stream, dist_epilogue);
//     } else {
//       THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
//     }

//     update_and_sample_thread.join();
//     // at this point:
//     // Old neighbors have been resampled on CPU (sample_graph(false)).
//     // New candidates from GPU are in graph_buffer_ and dists_buffer_.

//     if (update_counter_ == -1) { break; }
//     raft::copy(graph_host_buffer_.data_handle(),
//                graph_buffer_.data_handle(),
//                nrow_ * DEGREE_ON_DEVICE,
//                raft::resource::get_cuda_stream(res));
//     raft::resource::sync_stream(res);
//     raft::copy(dists_host_buffer_.data_handle(),
//                dists_buffer_.data_handle(),
//                nrow_ * DEGREE_ON_DEVICE,
//                raft::resource::get_cuda_stream(res));

//     // This is where the GPU-generated candidate neighbors are filtered:
//     graph_.sample_graph_new(graph_host_buffer_.data_handle(), DEGREE_ON_DEVICE);
//   }

//   graph_.update_graph(graph_host_buffer_.data_handle(),
//                       dists_host_buffer_.data_handle(),
//                       DEGREE_ON_DEVICE,
//                       update_counter_);
//   raft::resource::sync_stream(res);
//   graph_.sort_lists();

//   // Reuse graph_.h_dists as the buffer for shrink the lists in graph
//   static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));

//   if (return_distances) {
//     auto graph_h_dists = raft::make_host_matrix<DistData_t, int64_t, raft::row_major>(
//       nrow_, build_config_.output_graph_degree);

// // slice on host
// #pragma omp parallel for
//     for (size_t i = 0; i < (size_t)nrow_; i++) {
//       for (size_t j = 0; j < build_config_.output_graph_degree; j++) {
//         graph_h_dists(i, j) = graph_.h_dists(i, j);
//       }
//     }
//     raft::copy(output_distances,
//                graph_h_dists.data_handle(),
//                nrow_ * build_config_.output_graph_degree,
//                raft::resource::get_cuda_stream(res));

//     auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
//       output_distances, nrow_, build_config_.output_graph_degree);
//     // distance post-processing
//     bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;
//     if (build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
//         can_postprocess_dist) {
//       raft::linalg::map(
//         res, output_dist_view, raft::sqrt_op{}, raft::make_const_mdspan(output_dist_view));
//     } else if (!cuvs::distance::is_min_close(build_config_.metric) && can_postprocess_dist) {
//       // revert negated innerproduct
//       raft::linalg::map(res,
//                         output_dist_view,
//                         raft::mul_const_op<DistData_t>(-1),
//                         raft::make_const_mdspan(output_dist_view));
//     }
//     raft::resource::sync_stream(res);
//   }

//   Index_t* graph_shrink_buffer = (Index_t*)graph_.h_dists.data_handle();

// #pragma omp parallel for
//   for (size_t i = 0; i < (size_t)nrow_; i++) {
//     for (size_t j = 0; j < build_config_.node_degree; j++) {
//       size_t idx = i * graph_.node_degree + j;
//       int id     = graph_.h_graph[idx].id();
//       if (id < static_cast<int>(nrow_)) {
//         graph_shrink_buffer[i * build_config_.node_degree + j] = id;
//       } else {
//         graph_shrink_buffer[i * build_config_.node_degree + j] =
//           cuvs::neighbors::cagra::detail::device::xorshift64(idx) % nrow_;
//       }
//     }
//   }
//   graph_.h_graph = nullptr;

// #pragma omp parallel for
//   for (size_t i = 0; i < (size_t)nrow_; i++) {
//     for (size_t j = 0; j < build_config_.node_degree; j++) {
//       output_graph[i * build_config_.node_degree + j] =
//         graph_shrink_buffer[i * build_config_.node_degree + j];
//     }
//   }
// }

// checked_flagged_heap_push for raw arrays
template <typename DistT, typename IndexT>
int checked_flagged_heap_push(
  DistT* priorities, IndexT* indices, uint8_t* flags, size_t size, DistT p, IndexT n, uint8_t f)
{
  if (p >= priorities[0]) return 0;

  for (size_t i = 0; i < size; ++i)
    if (indices[i] == n) return 0;

  priorities[0] = p;
  indices[0]    = n;
  flags[0]      = f;

  size_t i = 0;
  while (true) {
    size_t ic1 = 2 * i + 1;
    size_t ic2 = ic1 + 1;
    size_t i_swap;

    if (ic1 >= size)
      break;
    else if (ic2 >= size) {
      if (priorities[ic1] > p)
        i_swap = ic1;
      else
        break;
    } else if (priorities[ic1] >= priorities[ic2]) {
      if (p < priorities[ic1])
        i_swap = ic1;
      else
        break;
    } else {
      if (p < priorities[ic2])
        i_swap = ic2;
      else
        break;
    }

    priorities[i] = priorities[i_swap];
    indices[i]    = indices[i_swap];
    flags[i]      = flags[i_swap];

    i = i_swap;
  }

  priorities[i] = p;
  indices[i]    = n;
  flags[i]      = f;

  return 1;
}

template <typename DistT, typename IndexT>
int checked_heap_push(DistT* priorities, IndexT* indices, DistT p, IndexT n, int size)
{
  if (p >= priorities[0]) return 0;

  for (int i = 0; i < size; ++i)
    if (indices[i] == n) return 0;

  priorities[0] = p;
  indices[0]    = n;

  int i = 0;
  while (true) {
    int ic1    = 2 * i + 1;
    int ic2    = ic1 + 1;
    int i_swap = -1;

    if (ic1 >= size)
      break;
    else if (ic2 >= size) {
      if (priorities[ic1] > p)
        i_swap = ic1;
      else
        break;
    } else if (priorities[ic1] >= priorities[ic2]) {
      if (p < priorities[ic1])
        i_swap = ic1;
      else
        break;
    } else {
      if (p < priorities[ic2])
        i_swap = ic2;
      else
        break;
    }

    std::swap(priorities[i], priorities[i_swap]);
    std::swap(indices[i], indices[i_swap]);
    i = i_swap;
  }

  priorities[i] = p;
  indices[i]    = n;
  return 1;
}

// Euclidean distance between two points
template <typename Data_t>
Data_t euclidean_distance(const Data_t* a, const Data_t* b, size_t dim)
{
  float sum = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

// init_random using pointers
template <typename Data_t, typename DistT, typename IndexT>
void init_random(size_t n_neighbors,
                 const Data_t* data,  // flattened: n_vertices * dim
                 size_t n_vertices,
                 size_t dim,
                 DistT* heap_priorities,  // flattened: n_vertices * n_neighbors
                 IndexT* heap_indices,    // flattened: n_vertices * n_neighbors
                 uint8_t* heap_flags      // flattened: n_vertices * n_neighbors
)
{
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<size_t> dist_idx(0, n_vertices - 1);

  for (size_t i = 0; i < n_vertices; ++i) {
    if (heap_indices[i * n_neighbors] < 0) {
      size_t n_valid = 0;
      for (size_t j = 0; j < n_neighbors; ++j)
        if (heap_indices[i * n_neighbors + j] >= 0) n_valid++;

      size_t n_to_add = n_neighbors - n_valid;

      for (size_t j = 0; j < n_to_add; ++j) {
        size_t idx = dist_idx(rng);

        DistT d = euclidean_distance(data + idx * dim, data + i * dim, dim);

        checked_flagged_heap_push(heap_priorities + i * n_neighbors,
                                  heap_indices + i * n_neighbors,
                                  heap_flags + i * n_neighbors,
                                  n_neighbors,
                                  d,
                                  static_cast<IndexT>(idx),
                                  static_cast<uint8_t>(1));
      }
    }
  }
}

template <typename DistT, typename IndexT>
void new_build_candidates(IndexT* current_indices,
                          uint8_t* current_flags,
                          int n_vertices,
                          int n_neighbors,
                          int max_candidates,
                          IndexT* new_candidate_indices,
                          DistT* new_candidate_priority,
                          IndexT* old_candidate_indices,
                          DistT* old_candidate_priority)
{
  // Random generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < n_vertices * max_candidates; ++i) {
    new_candidate_indices[i]  = -1;
    new_candidate_priority[i] = std::numeric_limits<DistT>::infinity();
    old_candidate_indices[i]  = -1;
    old_candidate_priority[i] = std::numeric_limits<DistT>::infinity();
  }

  // Iterate over vertices and neighbors sequentially
  for (int i = 0; i < n_vertices; ++i) {
    for (int j = 0; j < n_neighbors; ++j) {
      int idx     = current_indices[i * n_neighbors + j];
      uint8_t isn = current_flags[i * n_neighbors + j];

      if (idx < 0) continue;

      float d = dist(gen);  // random priority

      if (isn) {
        checked_heap_push(&new_candidate_priority[i * max_candidates],
                          &new_candidate_indices[i * max_candidates],
                          d,
                          idx,
                          max_candidates);
        checked_heap_push(&new_candidate_priority[idx * max_candidates],
                          &new_candidate_indices[idx * max_candidates],
                          d,
                          i,
                          max_candidates);
      } else {
        checked_heap_push(&old_candidate_priority[i * max_candidates],
                          &old_candidate_indices[i * max_candidates],
                          d,
                          idx,
                          max_candidates);
        checked_heap_push(&old_candidate_priority[idx * max_candidates],
                          &old_candidate_indices[idx * max_candidates],
                          d,
                          i,
                          max_candidates);
      }
    }
  }

  // Update flags for new candidates
  for (int i = 0; i < n_vertices; ++i) {
    for (int j = 0; j < n_neighbors; ++j) {
      int idx = current_indices[i * n_neighbors + j];
      for (int k = 0; k < max_candidates; ++k) {
        if (new_candidate_indices[i * max_candidates + k] == idx) {
          current_flags[i * n_neighbors + j] = static_cast<uint8_t>(0);
          break;
        }
      }
    }
  }
}

// Generate graph updates
template <typename Data_t, typename DistT, typename IndexT>
int process_graph_updates(IndexT* current_indices,    // [n_vertices * n_neighbors]
                          DistT* current_priorities,  // [n_vertices * n_neighbors]
                          uint8_t* current_flags,     // [n_vertices * n_neighbors]
                          IndexT* new_candidates,     // [n_vertices * max_candidates]
                          IndexT* old_candidates,     // [n_vertices * max_candidates]
                          DistT* dist_thresholds,     // [n_vertices]
                          const Data_t* data,         // [n_vertices * n_features]
                          size_t n_vertices,
                          size_t n_neighbors,
                          size_t max_candidates,
                          size_t n_features)
{
  int count = 0;
  for (size_t i = 0; i < n_vertices; ++i) {
    const Data_t* data_i = data + i * n_features;

    // Loop over new candidates
    for (size_t j = 0; j < max_candidates; ++j) {
      int p = new_candidates[i * max_candidates + j];
      if (p < 0) continue;

      const Data_t* data_p = data + p * n_features;

      // Compare with other new candidates
      for (size_t k = j; k < max_candidates; ++k) {
        int q = new_candidates[i * max_candidates + k];
        if (q < 0) continue;
        const Data_t* data_q = data + q * n_features;
        DistT d              = euclidean_distance(data_p, data_q, n_features);

        if (d <= dist_thresholds[p] || d <= dist_thresholds[q]) {
          count += checked_flagged_heap_push(current_priorities + p * n_neighbors,
                                             current_indices + p * n_neighbors,
                                             current_flags + p * n_neighbors,
                                             n_neighbors,
                                             d,
                                             q,
                                             1);
          count += checked_flagged_heap_push(current_priorities + q * n_neighbors,
                                             current_indices + q * n_neighbors,
                                             current_flags + q * n_neighbors,
                                             n_neighbors,
                                             d,
                                             p,
                                             1);
        }
      }

      // Compare with old candidates
      for (size_t k = 0; k < max_candidates; ++k) {
        int q = old_candidates[i * max_candidates + k];
        if (q < 0) continue;
        const Data_t* data_q = data + q * n_features;
        DistT d              = euclidean_distance(data_p, data_q, n_features);

        if (d <= dist_thresholds[p] || d <= dist_thresholds[q]) {
          count += checked_flagged_heap_push(current_priorities + p * n_neighbors,
                                             current_indices + p * n_neighbors,
                                             current_flags + p * n_neighbors,
                                             n_neighbors,
                                             d,
                                             q,
                                             1);
          count += checked_flagged_heap_push(current_priorities + q * n_neighbors,
                                             current_indices + q * n_neighbors,
                                             current_flags + q * n_neighbors,
                                             n_neighbors,
                                             d,
                                             p,
                                             1);
        }
      }
    }
  }
  return count;
}

template <typename DistT, typename IndexT>
void siftdown(DistT* heap1, IndexT* heap2, size_t size, size_t elt)
{
  while (elt * 2 + 1 < size) {
    size_t left_child  = elt * 2 + 1;
    size_t right_child = left_child + 1;
    size_t swap        = elt;

    if (heap1[swap] < heap1[left_child]) swap = left_child;

    if (right_child < size && heap1[swap] < heap1[right_child]) swap = right_child;

    if (swap == elt) break;

    std::swap(heap1[elt], heap1[swap]);
    std::swap(heap2[elt], heap2[swap]);

    elt = swap;
  }
}

template <typename DistT, typename IndexT>
void deheap_sort(IndexT* indices, DistT* distances, size_t n_rows, size_t n_neighbors)
{
  for (size_t row = 0; row < n_rows; ++row) {
    IndexT* row_indices  = indices + row * n_neighbors;
    DistT* row_distances = distances + row * n_neighbors;

    // Iterate from end of row to start
    for (size_t j = n_neighbors - 1; j > 0; --j) {
      // Swap root with last element in current heap
      std::swap(row_indices[0], row_indices[j]);
      std::swap(row_distances[0], row_distances[j]);

      // Siftdown on reduced heap
      siftdown(row_distances, row_indices, j, 0);
    }
  }
}

// template <typename Data_t, typename Index_t>
// template <typename DistEpilogue_t>
// void GNND<Data_t, Index_t>::build(Data_t* data,
//                                   const Index_t nrow,
//                                   Index_t* output_graph,  // int
//                                   bool return_distances,
//                                   DistData_t* output_distances,
//                                   DistEpilogue_t dist_epilogue)
// {
//   std::cout << "nrow is " << nrow << " build_config_.dataset_dim " << build_config_.dataset_dim
//             << " node degree " << build_config_.node_degree << std::endl;
//   using input_t = typename std::remove_const<Data_t>::type;
//   if (std::is_same_v<input_t, float>) {
//     std::vector<input_t> data_h(nrow * build_config_.dataset_dim);
//     int max_candidates = 30;
//     auto n_neighbors   = build_config_.node_degree;

//     std::vector<Index_t> indices(nrow * n_neighbors, -1);
//     std::vector<DistData_t> distances(nrow * n_neighbors,
//                                       std::numeric_limits<DistData_t>::infinity());
//     std::vector<uint8_t> flag(nrow * n_neighbors, 0);

//     init_random(n_neighbors,
//                 data,
//                 nrow,
//                 build_config_.dataset_dim,
//                 distances.data(),
//                 indices.data(),
//                 flag.data());

//     raft::print_host_vector("distances", distances.data(), 10, std::cout);
//     raft::print_host_vector("indices", indices.data(), 10, std::cout);

//     std::vector<Index_t> new_candidate_indices(nrow * max_candidates, -1);
//     std::vector<DistData_t> new_candidate_priority(nrow * max_candidates,
//                                                    std::numeric_limits<DistData_t>::infinity());
//     std::vector<Index_t> old_candidate_indices(nrow * max_candidates, -1);
//     std::vector<DistData_t> old_candidate_priority(nrow * max_candidates,
//                                                    std::numeric_limits<DistData_t>::infinity());

//     std::vector<DistData_t> dist_thresholds(nrow, 0);

//     for (size_t iter = 0; iter < build_config_.max_iterations; iter++) {
//       std::cout << "running iter " << iter << std::endl;
//       new_build_candidates(indices.data(),
//                            flag.data(),
//                            nrow,
//                            n_neighbors,
//                            max_candidates,
//                            new_candidate_indices.data(),
//                            new_candidate_priority.data(),
//                            old_candidate_indices.data(),
//                            old_candidate_priority.data());

//       // make dist_thresholds

//       for (int i = 0; i < nrow; ++i) {
//         const DistData_t* heap_i = distances.data() + i * n_neighbors;
//         // Since this is a max heap, the largest distance is at index 0
//         dist_thresholds[i] = heap_i[0];
//       }

//       int num_updated = process_graph_updates(indices.data(),
//                                               distances.data(),
//                                               flag.data(),
//                                               new_candidate_indices.data(),
//                                               old_candidate_indices.data(),
//                                               dist_thresholds.data(),
//                                               data,
//                                               nrow,
//                                               n_neighbors,
//                                               max_candidates,
//                                               build_config_.dataset_dim);
//       std::cout << "\tnum_updated " << num_updated << std::endl;
//       if (num_updated < 0.001 * n_neighbors * nrow) { break; }
//     }
//     std::cout << "done running!!!\n";

//     deheap_sort(indices.data(), distances.data(), nrow, n_neighbors);

//     raft::copy(
//       output_graph, indices.data(), nrow * n_neighbors, raft::resource::get_cuda_stream(res));

//     raft::print_host_vector("indices result in nnd", indices.data(), n_neighbors, std::cout);
//     raft::print_host_vector("indices distances in nnd", distances.data(), n_neighbors,
//     std::cout);

//     std::vector<DistData_t> distances_trim(nrow * build_config_.output_graph_degree);
//     for (int i = 0; i < nrow; i++) {
//       for (size_t j = 0; j < build_config_.output_graph_degree; j++) {
//         distances_trim[i * build_config_.output_graph_degree + j] = distances[i * n_neighbors +
//         j];
//       }
//     }
//     raft::copy(output_distances,
//                distances_trim.data(),
//                nrow * build_config_.output_graph_degree,
//                raft::resource::get_cuda_stream(res));

//   } else {
//     std::cout << "unsupported\n";
//   }
// }

template <typename Index_t>
void print_host_id_vector(const std::string& name,
                          const InternalID_t<Index_t>* data,
                          size_t n_elems,
                          std::ostream& os = std::cout)
{
  os << name << " [";
  for (size_t i = 0; i < n_elems; i++) {
    os << data[i].id();
    if (i + 1 < n_elems) os << ", ";
  }
  os << "]\n";
}

inline void print_host_int2_vector(const std::string& name,
                                   const int2* data,
                                   size_t n_elems,
                                   int dim,  // 0 for x, 1 for y
                                   std::ostream& os = std::cout)
{
  os << name << " [";
  for (size_t i = 0; i < n_elems; i++) {
    int val = (dim == 0 ? data[i].x : data[i].y);
    os << val;
    if (i + 1 < n_elems) os << ", ";
  }
  os << "]\n";
}

// try using the local join kernel
template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::build(Data_t* data,
                                  const Index_t nrow,
                                  Index_t* output_graph,  // int
                                  bool return_distances,
                                  DistData_t* output_distances,
                                  DistEpilogue_t dist_epilogue)
{
  using input_t = typename std::remove_const<Data_t>::type;
  if (build_config_.metric == cuvsDistanceType::BitwiseHamming &&
      !(std::is_same_v<input_t, uint8_t> || std::is_same_v<input_t, int8_t>)) {
    RAFT_FAIL(
      "Data type needs to be int8 or uint8 for NN Descent to run with BitwiseHamming distance.");
  }

  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  nrow_               = nrow;
  graph_.nrow         = nrow;
  graph_.bloom_filter.set_nrow(nrow);
  update_counter_ = 0;
  graph_.h_graph  = (InternalID_t<Index_t>*)output_graph;
  raft::matrix::fill(res, d_data_.view(), static_cast<float>(0));

  cudaPointerAttributes data_ptr_attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
  size_t batch_size = (data_ptr_attr.devicePointer == nullptr) ? 100000 : nrow_;

  cuvs::spatial::knn::detail::utils::batch_load_iterator vec_batches{
    data, static_cast<size_t>(nrow_), build_config_.dataset_dim, batch_size, stream};
  for (auto const& batch : vec_batches) {
    preprocess_data_kernel<<<
      batch.size(),
      raft::warp_size(),
      sizeof(Data_t) * ceildiv(build_config_.dataset_dim, static_cast<size_t>(raft::warp_size())) *
        raft::warp_size(),
      stream>>>(batch.data(),
                d_data_.data_handle(),
                build_config_.dataset_dim,
                l2_norms_.data_handle(),
                batch.offset(),
                build_config_.metric);
  }

  cudaDeviceSynchronize();
  // raft::print_device_vector("data0", d_data_.data_handle(), build_config_.dataset_dim,
  // std::cout); raft::print_device_vector("data1",
  //                           d_data_.data_handle() + build_config_.dataset_dim,
  //                           build_config_.dataset_dim,
  //                           std::cout);
  int target_idx = 0;
  if (std::is_same_v<input_t, float>) {
    // std::vector<input_t> data_h(nrow * build_config_.dataset_dim);
    auto n_neighbors = build_config_.node_degree;

    graph_.init_random_graph();
    // graph_.h_graph holds segment-based random selected indices now
    // graph_.h_dists also holds float max values

    // raft::print_host_vector("initialized dist for 2",
    //                         graph_.h_dists.data_handle() + 2 * n_neighbors,
    //                         n_neighbors,
    //                         std::cout);
    // print_host_id_vector(
    //   "initialized idx for 2", graph_.h_graph + 2 * n_neighbors, n_neighbors, std::cout);

    for (size_t iter = 0; iter < build_config_.max_iterations; iter++) {
      std::cout << "\nrunning iter " << iter << std::endl;
      graph_.sample_graph(true);
      // we have new and old for forward edges at this point
      // h_graph_old and h_graph_new is filled with forward edges
      // h_list_sizes_new.x h_list_sizes_old.x is filled with corresponding sizes

      raft::print_host_vector("h_graph_new after sampling",
                              graph_.h_graph_new.data_handle() + target_idx * NUM_SAMPLES,
                              NUM_SAMPLES,
                              std::cout);
      raft::print_host_vector("h_graph_old after sampling",
                              graph_.h_graph_old.data_handle() + target_idx * NUM_SAMPLES,
                              NUM_SAMPLES,
                              std::cout);
      print_host_int2_vector(
        "h_list_sizes_new fwd edge", graph_.h_list_sizes_new.data_handle(), 10, 0, std::cout);
      print_host_int2_vector(
        "h_list_sizes_old fwd edge", graph_.h_list_sizes_old.data_handle(), 10, 0, std::cout);

      // copy updated sizes
      raft::copy(
        d_list_sizes_new_.data_handle(), graph_.h_list_sizes_new.data_handle(), nrow_, stream);
      raft::copy(
        d_list_sizes_old_.data_handle(), graph_.h_list_sizes_old.data_handle(), nrow_, stream);

      // make reverse edges
      add_reverse_edges(graph_.h_graph_new.data_handle(),       // input
                        h_rev_graph_new_.data_handle(),         // final output
                        (Index_t*)dists_buffer_.data_handle(),  // tmp storage
                        d_list_sizes_new_.data_handle(),
                        stream);
      add_reverse_edges(graph_.h_graph_old.data_handle(),       // input
                        h_rev_graph_old_.data_handle(),         // final output
                        (Index_t*)dists_buffer_.data_handle(),  // tmp storage
                        d_list_sizes_old_.data_handle(),
                        stream);
      cudaDeviceSynchronize();

      raft::print_host_vector("h_rev_graph_new_ after reverse",
                              h_rev_graph_new_.data_handle() + target_idx * NUM_SAMPLES,
                              NUM_SAMPLES,
                              std::cout);
      raft::print_host_vector("h_rev_graph_old_ after reverse",
                              h_rev_graph_old_.data_handle() + target_idx * NUM_SAMPLES,
                              NUM_SAMPLES,
                              std::cout);

      // at this point d_list_sizes_new_ and d_list_sizes_old_ have the proper values of all list
      // sizes i.e. .x for forward edges and .y for reverse edges size graph_.h_graph_old and
      // graph_.h_graph_new has forward candidates h_rev_graph_old_ and h_rev_graph_new_ has reverse
      // candidates

      auto kernel       = preprocess_data_kernel<input_t>;
      void* kernel_ptr  = reinterpret_cast<void*>(kernel);
      auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
      auto wmma_range =
        raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

      if (wmma_range.contains(runtime_arch)) {
        local_join(stream, dist_epilogue);
      } else {
        THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
      }

      cudaDeviceSynchronize();
      // at this point we have valid sample results in graph_buffer_ and dists_buffer_

      raft::copy(graph_host_buffer_.data_handle(),
                 graph_buffer_.data_handle(),
                 nrow_ * DEGREE_ON_DEVICE,
                 raft::resource::get_cuda_stream(res));
      raft::copy(dists_host_buffer_.data_handle(),
                 dists_buffer_.data_handle(),
                 nrow_ * DEGREE_ON_DEVICE,
                 raft::resource::get_cuda_stream(res));

      cudaDeviceSynchronize();
      print_host_id_vector("graph_host_buffer_ after local join",
                           graph_host_buffer_.data_handle() + target_idx * DEGREE_ON_DEVICE,
                           DEGREE_ON_DEVICE,
                           std::cout);
      raft::print_host_vector("dists_host_buffer_ after local join",
                              dists_host_buffer_.data_handle() + target_idx * DEGREE_ON_DEVICE,
                              DEGREE_ON_DEVICE,
                              std::cout);

      // now we have valid sample results on host mirrors
      // we have to now write it back to proper places in h_graph and h_dists

      update_counter_ = 0;
      graph_.update_graph(graph_host_buffer_.data_handle(),
                          dists_host_buffer_.data_handle(),
                          DEGREE_ON_DEVICE,
                          update_counter_);
      cudaDeviceSynchronize();
      // at this point we have updated h_graph and h_dists (segmented)
      raft::print_host_vector("updated dist",
                              graph_.h_dists.data_handle() + target_idx * n_neighbors,
                              n_neighbors,
                              std::cout);
      print_host_id_vector(
        "updated idx", graph_.h_graph + target_idx * n_neighbors, n_neighbors, std::cout);

      std::cout << "num updates: " << update_counter_ << std::endl;
      if (update_counter_ < 0.0001 * n_neighbors * nrow) { break; }
    }

    graph_.sort_lists();

    std::vector<DistData_t> distances_trim(nrow * build_config_.output_graph_degree);
    for (int i = 0; i < nrow; i++) {
      for (size_t j = 0; j < build_config_.output_graph_degree; j++) {
        distances_trim[i * build_config_.output_graph_degree + j] =
          graph_.h_dists.data_handle()[i * n_neighbors + j];
      }
    }
    raft::copy(output_distances,
               distances_trim.data(),
               nrow * build_config_.output_graph_degree,
               raft::resource::get_cuda_stream(res));

    auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);
    // distance post-processing
    bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;
    if (build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
        can_postprocess_dist) {
      raft::linalg::map(
        res, output_dist_view, raft::sqrt_op{}, raft::make_const_mdspan(output_dist_view));
    } else if (!cuvs::distance::is_min_close(build_config_.metric) && can_postprocess_dist) {
      // revert negated innerproduct
      raft::linalg::map(res,
                        output_dist_view,
                        raft::mul_const_op<DistData_t>(-1),
                        raft::make_const_mdspan(output_dist_view));
    }

    Index_t* graph_shrink_buffer = (Index_t*)graph_.h_dists.data_handle();

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)nrow_; i++) {
      for (size_t j = 0; j < build_config_.node_degree; j++) {
        size_t idx = i * graph_.node_degree + j;
        int id     = graph_.h_graph[idx].id();
        if (id < static_cast<int>(nrow_)) {
          graph_shrink_buffer[i * build_config_.node_degree + j] = id;
        } else {
          graph_shrink_buffer[i * build_config_.node_degree + j] =
            cuvs::neighbors::cagra::detail::device::xorshift64(idx) % nrow_;
        }
      }
    }
    graph_.h_graph = nullptr;

#pragma omp parallel for
    for (size_t i = 0; i < (size_t)nrow_; i++) {
      for (size_t j = 0; j < build_config_.node_degree; j++) {
        output_graph[i * build_config_.node_degree + j] =
          graph_shrink_buffer[i * build_config_.node_degree + j];
      }
    }

  } else {
    std::cout << "unsupported\n";
  }
}

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
void build(raft::resources const& res,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
           index<IdxT>& idx)
{
  size_t extended_graph_degree, graph_degree;
  auto build_config = get_build_config(res,
                                       params,
                                       static_cast<size_t>(dataset.extent(0)),
                                       static_cast<size_t>(dataset.extent(1)),
                                       idx.metric(),
                                       extended_graph_degree,
                                       graph_degree);

  auto int_graph =
    raft::make_host_matrix<int, int64_t, raft::row_major>(dataset.extent(0), extended_graph_degree);
  // std::cout << "extended graph degree " << extended_graph_degree << std::endl;
  GNND<const T, int> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset.data_handle(),
              dataset.extent(0),
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle());
  } else {
    RAFT_EXPECTS(!params.return_distances,
                 "Distance view not allocated. Using return_distances set to true requires "
                 "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      auto graph                  = idx.graph().data_handle();
      graph[i * graph_degree + j] = int_graph.data_handle()[i * extended_graph_degree + j];
    }
  }
}

template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  index<IdxT> idx{res,
                  dataset.extent(0),
                  static_cast<int64_t>(graph_degree),
                  params.return_distances,
                  params.metric};

  build(res, params, dataset, idx);

  return idx;
}

}  // namespace cuvs::neighbors::nn_descent::detail
