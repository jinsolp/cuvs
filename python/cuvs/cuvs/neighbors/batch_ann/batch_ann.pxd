#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: language_level=3

from libc.stdint cimport uint32_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType


cdef extern from "cuvs/neighbors/batch_ann.h" nogil:
    ctypedef enum cuvsBatchANNGraphBuildAlgo:
        IVF_PQ
        NN_DESCENT

    ctypedef struct cuvsBatchANNIndexParams:
        cuvsDistanceType metric
        float metric_arg
        cuvsBatchANNGraphBuildAlgo build_algo
        size_t n_nearest_clusters
        size_t n_clusters
        size_t k

    ctypedef cuvsBatchANNIndexParams* cuvsBatchANNIndexParams_t

    ctypedef struct cuvsBatchANNIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsBatchANNIndex* cuvsBatchANNIndex_t

    cuvsError_t cuvsBatchANNIndexParamsCreate(
        cuvsBatchANNIndexParams_t* params)

    cuvsError_t cuvsBatchANNIndexParamsDestroy(
        cuvsBatchANNIndexParams_t index)

    # cuvsError_t cuvsNNDescentIndexCreate(cuvsNNDescentIndex_t* index)

    # cuvsError_t cuvsNNDescentIndexDestroy(cuvsNNDescentIndex_t index)

    # cuvsError_t cuvsNNDescentIndexGetGraph(cuvsNNDescentIndex_t index,
    #                                        DLManagedTensor * output)

    cuvsError_t cuvsBatchANNBuildClusters(cuvsResources_t res,
                                          cuvsBatchANNIndexParams* params,
                                          DLManagedTensor* dataset,
                                          size_t k,
                                          size_t* max_cluster_size,
                                          size_t* min_cluster_size,
                                          DLManagedTensor* cluster_sizes,
                                          DLManagedTensor* cluster_offsets,
                                          DLManagedTensor* inverted_indices) \
        except +
