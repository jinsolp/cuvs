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

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/copy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/batch_ann.h>
#include <cuvs/neighbors/batch_ann.hpp>

#include <fstream>

namespace {

template <typename T, typename IdxT = uint32_t>
void _build_clusters(cuvsResources_t res,
                     cuvsBatchANNIndexParams params,
                     DLManagedTensor* dataset_tensor,
                     size_t k,
                     size_t& max_cluster_size,
                     size_t& min_cluster_size,
                     DLManagedTensor* cluster_sizes,
                     DLManagedTensor* cluster_offsets,
                     DLManagedTensor* inverted_indices)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto dataset = dataset_tensor->dl_tensor;

  auto build_params               = cuvs::neighbors::batch_ann::index_params();
  build_params.metric             = static_cast<cuvs::distance::DistanceType>((int)params.metric),
  build_params.metric_arg         = params.metric_arg;
  build_params.n_nearest_clusters = params.n_nearest_clusters;
  build_params.n_clusters         = params.n_clusters;

  switch (params.build_algo) {
    case cuvsBatchANNGraphBuildAlgo::IVF_PQ: {
      cuvs::neighbors::batch_ann::graph_build_params::ivf_pq_params ivf_pq_params{};
      ivf_pq_params.build_params.n_lists =
        std::max(5u,
                 static_cast<uint32_t>(dataset.shape[0] * params.n_nearest_clusters /
                                       (5000 * params.n_clusters)));
      build_params.graph_build_params = ivf_pq_params;
      break;
    }
    case cuvsBatchANNGraphBuildAlgo::NN_DESCENT: {
      cuvs::neighbors::batch_ann::graph_build_params::nn_descent_params nn_descent_params{};
      nn_descent_params.max_iterations            = 1000;
      nn_descent_params.graph_degree              = params.k;
      nn_descent_params.intermediate_graph_degree = params.k * 2;
      build_params.graph_build_params             = nn_descent_params;
      break;
    }
  };

  using dataset_type           = raft::host_matrix_view<T const, int64_t, raft::row_major>;
  auto dataset_to_pass         = cuvs::core::from_dlpack<dataset_type>(dataset_tensor);
  using vector_type            = raft::host_vector_view<int64_t, int64_t>;
  auto cluster_sizes_mdspan    = cuvs::core::from_dlpack<vector_type>(cluster_sizes);
  auto cluster_offsets_mdspan  = cuvs::core::from_dlpack<vector_type>(cluster_offsets);
  auto inverted_indices_mdspan = cuvs::core::from_dlpack<vector_type>(inverted_indices);
  cuvs::neighbors::batch_ann::build_clusters(*res_ptr,
                                             dataset_to_pass,
                                             build_params,
                                             k,
                                             max_cluster_size,
                                             min_cluster_size,
                                             cluster_sizes_mdspan,
                                             cluster_offsets_mdspan,
                                             inverted_indices_mdspan);
}
}  // namespace

// extern "C" cuvsError_t cuvsBatchANNIndexCreate(cuvsBatchANNIndex_t* index)
// {
//   return cuvs::core::translate_exceptions([=] { *index = necuvsBatchANNIndex{}; });
// }

// extern "C" cuvsError_t cuvsBatchANNIndexDestroy(cuvsBatchANNIndex_t index_c_ptr)
// {
//   return cuvs::core::translate_exceptions([=] {
//     auto index = *index_c_ptr;
//     if ((index.dtype.code == kDLUInt) && (index.dtype.bits == 32)) {
//       auto index_ptr =
//       reinterpret_cast<cuvs::neighbors::nn_descent::index<uint32_t>*>(index.addr); delete
//       index_ptr;
//     } else {
//       RAFT_FAIL(
//         "Unsupported nn-descent index dtype: %d and bits: %d", index.dtype.code,
//         index.dtype.bits);
//     }
//     delete index_c_ptr;
//   });
// }

extern "C" cuvsError_t cuvsBatchANNBuildClusters(cuvsResources_t res,
                                                 cuvsBatchANNIndexParams_t params,
                                                 DLManagedTensor* dataset_tensor,
                                                 size_t k,
                                                 size_t* max_cluster_size,
                                                 size_t* min_cluster_size,
                                                 DLManagedTensor* cluster_sizes,
                                                 DLManagedTensor* cluster_offsets,
                                                 DLManagedTensor* inverted_indices)
{
  return cuvs::core::translate_exceptions([=] {
    // index->dtype.code = kDLUInt;
    // index->dtype.bits = 32;

    auto dtype = dataset_tensor->dl_tensor.dtype;

    if ((dtype.code == kDLFloat) && (dtype.bits == 32)) {
      // should be going in here
      // index->addr = reinterpret_cast<uintptr_t>(
      size_t max_cluster_size, min_cluster_size;
      _build_clusters<float, int64_t>(res,
                                      *params,
                                      dataset_tensor,
                                      k,
                                      max_cluster_size,
                                      min_cluster_size,
                                      cluster_sizes,
                                      cluster_offsets,
                                      inverted_indices);
      // } else if ((dtype.code == kDLFloat) && (dtype.bits == 16)) {
      //   index->addr = reinterpret_cast<uintptr_t>(
      //     _build<half, uint32_t>(res, *params, dataset_tensor, graph_tensor));
      // } else if ((dtype.code == kDLInt) && (dtype.bits == 8)) {
      //   index->addr = reinterpret_cast<uintptr_t>(
      //     _build<int8_t, uint32_t>(res, *params, dataset_tensor, graph_tensor));
      // } else if ((dtype.code == kDLUInt) && (dtype.bits == 8)) {
      //   index->addr = reinterpret_cast<uintptr_t>(
      //     _build<uint8_t, uint32_t>(res, *params, dataset_tensor, graph_tensor));
    } else {
      RAFT_FAIL("Unsupported batch ann dataset dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsBatchANNIndexParamsCreate(cuvsBatchANNIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    // get defaults from cpp parameters struct
    cuvs::neighbors::batch_ann::index_params cpp_params;

    *params = new cuvsBatchANNIndexParams{.metric             = cpp_params.metric,
                                          .metric_arg         = cpp_params.metric_arg,
                                          .build_algo         = cuvsBatchANNGraphBuildAlgo::IVF_PQ,
                                          .n_nearest_clusters = cpp_params.n_nearest_clusters,
                                          .n_clusters         = cpp_params.n_clusters,
                                          .k                  = 32};
  });
}

extern "C" cuvsError_t cuvsBatchANNIndexParamsDestroy(cuvsBatchANNIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

// extern "C" cuvsError_t cuvsBatchANNIndexGetGraph(cuvsBatchANNIndex_t index,
//                                                   DLManagedTensor* graph)
// {
//   return cuvs::core::translate_exceptions([=] {
//     auto dtype = index->dtype;
//     if ((dtype.code == kDLUInt) && (dtype.bits == 32)) {
//       auto index_ptr =
//       reinterpret_cast<cuvs::neighbors::nn_descent::index<uint32_t>*>(index->addr); using
//       output_mdspan_type = raft::host_matrix_view<uint32_t, int64_t, raft::row_major>; auto dst
//       = cuvs::core::from_dlpack<output_mdspan_type>(graph); auto src                 =
//       index_ptr->graph();

//       RAFT_EXPECTS(src.extent(0) == dst.extent(0), "Output graph has incorrect number of rows");
//       RAFT_EXPECTS(src.extent(1) == dst.extent(1), "Output graph has incorrect number of cols");
//       std::copy(src.data_handle(), src.data_handle() + dst.size(), dst.data_handle());
//     } else {
//       RAFT_FAIL("Unsupported nn-descent index dtype: %d and bits: %d", dtype.code, dtype.bits);
//     }
//   });
// }
