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

import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import batch_ann, brute_force
from cuvs.tests.ann_utils import calc_recall

from sklearn.datasets import make_blobs
import h5py


def test_batch_ann(n_rows, n_cols):
    metric = "sqeuclidean"
    build_algo = "ivf_pq"
    n_nearest_clusters = 2
    n_clusters = 10
    k = 32


    hf = h5py.File("/home/coder/data/sift/sift_base.hdf5", 'r')
    input1 = np.array(hf['train'])
    n_rows = input1.shape[0]
    n_cols = input1.shape[1]
    print(type(input1))
    # input1 = make_blobs(n_samples=n_rows, centers=5, n_features=n_cols, random_state=42)[0].astype(np.float32)
    # input1 = np.random.random_sample((n_rows, n_cols)).astype(np.float32)
    
    input1_device = device_ndarray(input1)

    params = batch_ann.IndexParams(metric=metric, build_algo = build_algo,
            n_nearest_clusters = n_nearest_clusters, n_clusters = n_clusters,
            k = k)

    cluster_sizes = np.zeros((n_clusters), dtype="int64")
    cluster_offsets = np.zeros((n_clusters), dtype="int64")
    inverted_indices = np.zeros((n_rows * n_nearest_clusters), dtype="int64")
    batch_ann.build_clusters(
        params,
        input1,
        cluster_sizes,
        cluster_offsets,
        inverted_indices
    )
    print("cluster sizes:", cluster_sizes)
   
    max_cluster_size = np.max(cluster_sizes)
    min_cluster_size = np.min(cluster_sizes)
    print(min_cluster_size, max_cluster_size)

    idx = batch_ann.full_single_gpu_build(params,
                                           input1,
                                           max_cluster_size,
                                           min_cluster_size,
                                           n_clusters,
                                           cluster_sizes,
                                           cluster_offsets,
                                           inverted_indices)
    # if not inplace:
    #     graph = index.graph

    print(idx.graph.shape)
    # print(idx.graph)

    bfknn_index = brute_force.build(input1_device, metric=metric)
    _, bfknn_graph = brute_force.search(
        bfknn_index, input1_device, k=k
    )
    bfknn_graph = bfknn_graph.copy_to_host()

    print(f"recall: {calc_recall(idx.graph, bfknn_graph)}")
    # assert calc_recall(idx.graph, bfknn_graph) > 0.9


if __name__ == "__main__":
    n_rows = 100000
    n_cols = 128
    test_batch_ann(n_rows, n_cols)