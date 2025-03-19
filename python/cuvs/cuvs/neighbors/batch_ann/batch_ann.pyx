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

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from cython.operator cimport dereference as deref
from libcpp cimport bool, cast
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_NAMES, DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from libc.stdint cimport (
    int8_t,
    int64_t,
    uint8_t,
    uint32_t,
    uint64_t,
    uintptr_t,
)

from cuvs.common.exceptions import check_cuvs


cdef class IndexParams:
    """
    Parameters to build NN-Descent Index


    """

    cdef cuvsBatchANNIndexParams* params
    cdef object _metric

    def __cinit__(self):
        cuvsBatchANNIndexParamsCreate(&self.params)

    def __dealloc__(self):
        check_cuvs(cuvsBatchANNIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric="sqeuclidean",
                 build_algo="ivf_pq",
                 n_nearest_clusters=2,
                 n_clusters=4,
                 k=32):
        self._metric = metric
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.n_nearest_clusters = n_nearest_clusters
        self.params.n_clusters = n_clusters
        self.params.k = k

        # setting this parameter to true will cause an exception in the c++
        # api (`Using return_distances set to true requires distance view to
        # be allocated.`) - so instead force to be false here
        # self.params.return_distances = False

    @property
    def metric(self):
        return DISTANCE_NAMES[self.params.metric]

    @property
    def n_clusters(self):
        return self.params.n_clusters

cdef class Index:
    """
    NN-Descent index object. This object stores the trained NN-Descent index,
    which can be used to get the NN-Descent graph and distances after
    building
    """

    cdef cuvsBatchANNIndex_t index
    # cdef bool trained
    cdef int64_t num_rows
    cdef size_t k

    def __cinit__(self):
        # self.num_rows = 0
        # self.k = 0
        check_cuvs(cuvsBatchANNIndexCreate(&self.index))

    def __dealloc__(self):
        # check_cuvs(cuvsNNDescentIndexDestroy(self.index))
        pass

    def __init__(self, *,
                 num_rows=4,
                 k=32):
        # self._metric = metric
        # self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        # self.params.n_nearest_clusters = n_nearest_clusters
        # self.params.n_clusters = n_clusters
        # self.params.k = k
        self.num_rows = num_rows
        self.k = k

    @property
    def graph(self):

        output = np.empty((self.num_rows, self.k), dtype='int64')
        ai = wrap_array(output)
        cdef cydlpack.DLManagedTensor* output_dlpack = cydlpack.dlpack_c(ai)
        check_cuvs(cuvsBatchANNIndexGetGraph(self.index, output_dlpack))
        return output

    def __repr__(self):
        return "Index(type=BatchANN)"


@auto_sync_resources
def build_clusters(IndexParams index_params,
                   dataset,
                   cluster_sizes,
                   cluster_offsets,
                   inverted_indices,
                   resources=None):

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    # cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsBatchANNIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* graph_dlpack = NULL
    # if graph is not None:
    #     graph_ai = wrap_array(graph)
    #     graph_dlpack = cydlpack.dlpack_c(graph_ai)
    cluster_sizes_ai = wrap_array(cluster_sizes)
    cluster_sizes_dlpack = cydlpack.dlpack_c(cluster_sizes_ai)

    cluster_offsets_ai = wrap_array(cluster_offsets)
    cluster_offsets_dlpack = cydlpack.dlpack_c(cluster_offsets_ai)

    inverted_indices_ai = wrap_array(inverted_indices)
    inverted_indices_dlpack = cydlpack.dlpack_c(inverted_indices_ai)

    with cuda_interruptible():
        check_cuvs(cuvsBatchANNBuildClusters(
            res,
            params,
            dataset_dlpack,
            params.k,
            cluster_sizes_dlpack,
            cluster_offsets_dlpack,
            inverted_indices_dlpack
        ))

    return


@auto_sync_resources
def full_single_gpu_build(IndexParams index_params,
                          dataset,
                          max_cluster_size,
                          min_cluster_size,
                          n_clusters,
                          cluster_sizes,
                          cluster_offsets,
                          inverted_indices,
                          resources=None):

    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, [np.dtype('float32'), np.dtype('byte'),
                                    np.dtype('ubyte')])

    # cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsBatchANNIndexParams* params = index_params.params

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()

    cdef cydlpack.DLManagedTensor* graph_dlpack = NULL
    # if graph is not None:
    #     graph_ai = wrap_array(graph)
    #     graph_dlpack = cydlpack.dlpack_c(graph_ai)
    cluster_sizes_ai = wrap_array(cluster_sizes)
    cluster_sizes_dlpack = cydlpack.dlpack_c(cluster_sizes_ai)

    cluster_offsets_ai = wrap_array(cluster_offsets)
    cluster_offsets_dlpack = cydlpack.dlpack_c(cluster_offsets_ai)

    inverted_indices_ai = wrap_array(inverted_indices)
    inverted_indices_dlpack = cydlpack.dlpack_c(inverted_indices_ai)

    cdef Index idx = Index(num_rows=dataset_ai.shape[0], k=params.k)
    with cuda_interruptible():
        check_cuvs(cuvsBatchANNFullSingleGPUBuild(
            res,
            dataset_dlpack,
            max_cluster_size,
            min_cluster_size,
            n_clusters,
            params,
            cluster_sizes_dlpack,
            cluster_offsets_dlpack,
            inverted_indices_dlpack,
            idx.index
        ))

    return idx
