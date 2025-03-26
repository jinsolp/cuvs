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
import os
import pytest
import time
from pylibraft.common import device_ndarray


from cuvs.neighbors import batch_ann, brute_force, nn_descent
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
    print(idx.graph)

    bfknn_index = brute_force.build(input1_device, metric=metric)
    _, bfknn_graph = brute_force.search(
        bfknn_index, input1_device, k=k
    )
    bfknn_graph = bfknn_graph.copy_to_host()

    print(f"recall: {calc_recall(idx.graph, bfknn_graph)}")
    # assert calc_recall(idx.graph, bfknn_graph) > 0.9

def get_cpu_affinity(worker_id, n_workers, cores_per_worker):
    # Calculate the cores assigned to the worker
    start_core = worker_id * cores_per_worker
    end_core = start_core + cores_per_worker - 1
    return f"{start_core}-{end_core}"  # e.g., "0-31" for worker 0


def run_on_gpu(given_gpu_id, params, input1, max_cluster_size, min_cluster_size, n_clusters, cluster_sizes, cluster_offsets, inverted_indices, start_event, counter_var):
    # Here you can set the CUDA device, so the function runs on the correct GPU
    # import cupy as cp
    # import psutil
    # cpu_affinity = get_cpu_affinity(given_gpu_id, 4, 32)
    # psutil.Process(os.getpid()).cpu_affinity(list(range(int(cpu_affinity.split('-')[0]), int(cpu_affinity.split('-')[1]) + 1)))
    # print(f"Worker {given_gpu_id} bound to CPU cores: {cpu_affinity}")
    # gpu_id = dask.config.get("distributed.worker.resources.gpu", given_gpu_id)  # Get GPU ID for the worker
    # # print(f"here gpu id: ================{gpu_id}")
    # cp.cuda.Device(gpu_id).use()  # Set the current CUDA device
    
    # Now call your function with GPU support (assuming batch_ann is already imported)
    print(f"{given_gpu_id} started running run on gpu")
    count = counter_var.get() or 0
    counter_var.set(count + 1)
    print(f"[GPU {given_gpu_id}] Waiting... Current counter: {count + 1}/{4}")

    if count + 1 == 4:
        print(f"[GPU {given_gpu_id}] Last worker arrived! Setting start event...")
        start_event.set()  # Now all workers can start!
    else:
        start_event.wait()
    print(f"{given_gpu_id} we are here setting up")

    
    _ = batch_ann.full_single_gpu_build(params,
                                           input1,
                                           max_cluster_size,
                                           min_cluster_size,
                                           n_clusters,
                                           cluster_sizes,
                                           cluster_offsets,
                                           inverted_indices)
    
    return True


if __name__ == "__main__":
   
    from dask.distributed import Client, Event, Variable
    import dask
    from dask_cuda import LocalCUDACluster

    n_gpu = 4  # Example number of GPUs
    threads_per_worker =  int(32)
    dask.config.set({"distributed.nanny.pre-spawn-environ.OMP_NUM_THREADS": threads_per_worker})

    # os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    # worker_kwargs = {
    #     'env': {'OMP_NUM_THREADS': str(threads_per_worker)}
    # }

    # Launch the Dask cluster with worker kwargs to set environment variables
    # cluster = LocalCUDACluster(n_workers=4, CUDA_VISIBLE_DEVICES="1,2,3,4")
    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES="1,2,3,4")

    # cluster = LocalCUDACluster(n_workers=n_gpu)
    client = Client(cluster)
   

    metric = "sqeuclidean"
    build_algo = "nnd"
    n_nearest_clusters = 2
    n_clusters = 20
    k = 32

    # hf = h5py.File("/home/coder/data/sift/sift_base.hdf5", 'r')
    hf = h5py.File("/home/coder/data/deep_image/deep-image-96-angular.hdf5", 'r')
    
    input1 = np.array(hf['train'])
    n_rows = input1.shape[0]
    n_cols = input1.shape[1]
    print(n_rows, n_cols)
    # print(type(input1))

    # input1 = da.from_array(hf['train'], chunks=(n_rows, n_cols))

    
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
    print("cluster offsets:", cluster_offsets)
   
    max_cluster_size = np.max(cluster_sizes)
    min_cluster_size = np.min(cluster_sizes)
    print(min_cluster_size, max_cluster_size)

    # getting cluster splits per rank
    n_clusters_per_ranks = [n_clusters//n_gpu] * n_gpu
    rem = n_clusters%n_gpu
    n_clusters_per_ranks = [val + 1 if i < rem else val for i, val in enumerate(n_clusters_per_ranks)]
    rank_offsets = []
    print(n_clusters_per_ranks)
    num_data_per_ranks = []
    cluster_offsets_per_rank = []
    for i in range(n_gpu):
        base_cluster_idx = i * (n_clusters//n_gpu) + min(rem, i)
        rank_offsets.append(cluster_offsets[base_cluster_idx])
        # print(base_cluster_idx)
        rank_offset = cluster_offsets[base_cluster_idx]
        offsets_for_this_rank = []
        num_data_for_this_rank = 0
        for j in range(n_clusters_per_ranks[i]):
            offsets_for_this_rank.append(cluster_offsets[base_cluster_idx + j] - rank_offset)
            num_data_for_this_rank += cluster_sizes[base_cluster_idx + j]
        cluster_offsets_per_rank.append(offsets_for_this_rank)
        num_data_per_ranks.append(num_data_for_this_rank)
    print(cluster_offsets_per_rank)


    input1_future = client.scatter(input1)



    # params = nn_descent.IndexParams(metric=metric, graph_degree=k)
    start_event = Event(name="start_event")
    start_event_future = client.scatter(start_event)

    counter_var = Variable(name="worker_counter")
    counter_var.set(0)
    counter_var_future = client.scatter(counter_var)

    futures = []
    for gpu_id in range(n_gpu):
        base_cluster_idx = gpu_id * (n_clusters//n_gpu) + min(rem, gpu_id)
        print(base_cluster_idx)
        n_cluster_for_this_rank = n_clusters_per_ranks[gpu_id]
        print(f"gpu id {gpu_id} n cluster for this rank: {n_cluster_for_this_rank}\ncluster sizes: {cluster_sizes[base_cluster_idx:base_cluster_idx + n_cluster_for_this_rank]}, cluster offsets: {np.array(cluster_offsets_per_rank[gpu_id])}, inverted indices shape: {inverted_indices[rank_offsets[gpu_id] : rank_offsets[gpu_id] + num_data_per_ranks[gpu_id]].shape}")
        future = client.submit(run_on_gpu, gpu_id, params, input1_future, max_cluster_size, min_cluster_size, n_cluster_for_this_rank, 
                               cluster_sizes[base_cluster_idx:base_cluster_idx + n_cluster_for_this_rank], 
                               np.array(cluster_offsets_per_rank[gpu_id], dtype="int64"), 
                               inverted_indices[rank_offsets[gpu_id] : rank_offsets[gpu_id] + num_data_per_ranks[gpu_id]], 
                               start_event_future, counter_var_future)
        
        futures.append(future)

    results = client.gather(futures)
    print(results)

    client.close()