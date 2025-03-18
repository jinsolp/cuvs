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

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cuvsBatchANNGraphBuildAlgo {
  IVF_PQ,
  NN_DESCENT,
};

struct cuvsBatchANNIndexParams {
  cuvsDistanceType metric;
  float metric_arg;
  enum cuvsBatchANNGraphBuildAlgo build_algo;
  size_t n_nearest_clusters;
  size_t n_clusters;
  size_t k;  // additional k for NN Descent graph degree
};

typedef struct cuvsBatchANNIndexParams* cuvsBatchANNIndexParams_t;

cuvsError_t cuvsBatchANNIndexParamsCreate(cuvsBatchANNIndexParams_t* index_params);

cuvsError_t cuvsBatchANNIndexParamsDestroy(cuvsBatchANNIndexParams_t index_params);
/**
 * @}
 */

/**
 * @defgroup nn_descent_c_index NN-Descent index
 * @{
 */
/**
 * @brief Struct to hold address of cuvs::neighbors::nn_descent::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsBatchANNIndex;

typedef cuvsBatchANNIndex* cuvsBatchANNIndex_t;

/**
 * @brief Allocate NN-Descent index
 *
 * @param[in] index cuvsNNDescentIndex_t to allocate
 * @return cuvsError_t
 */
// cuvsError_t cuvsBatchANNIndexCreate(cuvsBatchANNIndex_t* index);

/**
 * @brief De-allocate NN-Descent index
 *
 * @param[in] index cuvsNNDescentIndex_t to de-allocate
 */
// cuvsError_t cuvsBatchANNIndexDestroy(cuvsBatchANNIndex_t index);
/**
 * @}
 */

/**
 * @defgroup nn_descent_c_index_build NN-Descent index build
 * @{
 */
/**
 * @brief Build a NN-Descent index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/nn_descent.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsNNDescentIndexParams_t index_params;
 * cuvsError_t params_create_status = cuvsNNDescentIndexParamsCreate(&index_params);
 *
 * // Create NN-Descent index
 * cuvsNNDescentIndex_t index;
 * cuvsError_t index_create_status = cuvsNNDescentIndexCreate(&index);
 *
 * // Build the NN-Descent Index
 * cuvsError_t build_status = cuvsNNDescentBuild(res, index_params, &dataset, index);
 *
 * // de-allocate `index_params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsNNDescentIndexParamsDestroy(index_params);
 * cuvsError_t index_destroy_status = cuvsNNDescentIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index_params cuvsNNDescentIndexParams_t used to build NN-Descent index
 * @param[in] dataset DLManagedTensor* training dataset on host or device memory
 * @param[inout] graph Optional preallocated graph on host memory to store output
 * @param[out] index cuvsNNDescentIndex_t Newly built NN-Descent index
 * @return cuvsError_t
 */
// this is for the single gpu ver
cuvsError_t cuvsBatchANNBuildClusters(cuvsResources_t res,
                                      cuvsBatchANNIndexParams_t params,
                                      DLManagedTensor* dataset_tensor,
                                      size_t k,
                                      size_t* max_cluster_size,
                                      size_t* min_cluster_size,
                                      DLManagedTensor* cluster_sizes,
                                      DLManagedTensor* cluster_offsets,
                                      DLManagedTensor* inverted_indices);
/**
 * @}
 */

/**
 * @brief Get the KNN graph from a built NN-Descent index
 *
 * @param[in] index cuvsNNDescentIndex_t Built NN-Descent index
 * @param[inout] graph Optional preallocated graph on host memory to store output
 * @return cuvsError_t
 */
// cuvsError_t cuvsBatchANNIndexGetGraph(cuvsBatchANNIndex_t index, DLManagedTensor* graph);
#ifdef __cplusplus
}
#endif
