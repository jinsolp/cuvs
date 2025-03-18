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

# cdef class Index:
#     """
#     NN-Descent index object. This object stores the trained NN-Descent index,
#     which can be used to get the NN-Descent graph and distances after
#     building
#     """

#     cdef cuvsNNDescentIndex_t index
#     cdef bool trained
#     cdef int64_t num_rows
#     cdef size_t graph_degree

#     def __cinit__(self):
#         self.trained = False
#         self.num_rows = 0
#         self.graph_degree = 0
#         check_cuvs(cuvsNNDescentIndexCreate(&self.index))

#     def __dealloc__(self):
#         check_cuvs(cuvsNNDescentIndexDestroy(self.index))

#     @property
#     def trained(self):
#         return self.trained

#     @property
#     def graph(self):
#         if not self.trained:
#             raise ValueError("Index needs to be built before getting graph")

#         output = np.empty((self.num_rows, self.graph_degree), dtype='uint32')
#         ai = wrap_array(output)
#         cdef cydlpack.DLManagedTensor* output_dlpack = cydlpack.dlpack_c(ai)
#         check_cuvs(cuvsNNDescentIndexGetGraph(self.index, output_dlpack))
#         return output

#     def __repr__(self):
#         return "Index(type=NNDescent)"


@auto_sync_resources
def build_clusters(IndexParams index_params,
                   dataset,
                   cluster_sizes,
                   cluster_offsets,
                   inverted_indices,
                   resources=None):
    """
    Build KNN graph from the dataset

    Parameters
    ----------
    index_params : :py:class:`cuvs.neighbors.nn_descent.IndexParams`
    dataset : Array interface compliant matrix, on either host or device memory
        Supported dtype [float, int8, uint8]
    graph : Optional host matrix for storing output graph
    {resources_docstring}

    Returns
    -------
    index: py:class:`cuvs.neighbors.nn_descent.Index`

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import nn_descent
    >>> n_samples = 50000
    >>> n_features = 50
    >>> n_queries = 1000
    >>> k = 10
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> build_params = nn_descent.IndexParams(metric="sqeuclidean")
    >>> index = nn_descent.build(build_params, dataset)
    >>> graph = index.graph
    """
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

    cdef size_t min_cluster_size, max_cluster_size
    with cuda_interruptible():
        check_cuvs(cuvsBatchANNBuildClusters(
            res,
            params,
            dataset_dlpack,
            params.k,
            &min_cluster_size,
            &max_cluster_size,
            cluster_sizes_dlpack,
            cluster_offsets_dlpack,
            inverted_indices_dlpack
        ))

    return min_cluster_size, min_cluster_size
