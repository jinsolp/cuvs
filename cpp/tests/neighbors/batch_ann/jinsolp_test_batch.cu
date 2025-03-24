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

//  #include "../test_utils.cuh"
// #include "../ann_utils.cuh"
#include <cstdint>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/nccl_clique.hpp>
// #include <raft/matrix/sample_rows.cuh>
#include <raft/random/make_blobs.cuh>
#include <sys/mman.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <rapids_logger/logger.hpp>
// #include <rmm/device_uvector.hpp>

// #include "naive_knn.cuh"
#include <cuvs/distance/distance.hpp>
// #include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/batch_ann.hpp>

#include <rmm/mr/device/statistics_resource_adaptor.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <vector>

enum knn_build_algo_ { NN_DESCENT, IVF_PQ };
enum data_gen_ { NORMAL, BLOBS, SIFT, GIST, OPENAI5M, DEEP, SIFT_MMAP, DEEP_MMAP, WIKI_MMAP, WIKI };
enum do_mmap_ { MMAP, NO_MMAP };

/** Calculate recall value using only neighbor indices
 */
template <typename T>
auto calc_recall(const std::vector<T>& expected_idx,
                 const std::vector<T>& actual_idx,
                 size_t rows,
                 size_t cols)
{
  size_t match_count = 0;
  size_t total_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t k = 0; k < cols; ++k) {
      size_t idx_k = i * cols + k;  // row major assumption!
      auto act_idx = actual_idx[idx_k];
      for (size_t j = 0; j < cols; ++j) {
        size_t idx   = i * cols + j;  // row major assumption!
        auto exp_idx = expected_idx[idx];
        if (act_idx == exp_idx) {
          match_count++;
          break;
        }
      }
    }
  }
  return std::make_tuple(
    static_cast<double>(match_count) / static_cast<double>(total_count), match_count, total_count);
}

template <typename T>
void eval_recall(const std::vector<T>& expected_idx,
                 const std::vector<T>& actual_idx,
                 size_t rows,
                 size_t cols,
                 double eps,
                 double min_recall,
                 bool test_unique = true)
{
  auto [actual_recall, match_count, total_count] =
    calc_recall(expected_idx, actual_idx, rows, cols);
  std::cout << "recall is " << actual_recall << std::endl;
}

template <typename T>
void write_vector_to_file(const std::vector<T>& vec, const std::string& filename)
{
  // Open file in binary mode
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error opening file for writing." << std::endl;
    return;
  }

  // Write the size of the vector
  size_t size = vec.size();
  out.write(reinterpret_cast<const char*>(&size), sizeof(size));

  // Write the vector data
  out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template <typename T>
std::vector<T> read_vector_from_file(const std::string& filename)
{
  // Open file in binary mode
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    std::cerr << "Error opening file for reading." << std::endl;
    return {};
  }

  // Read the size of the vector
  size_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(size));

  // Read the vector data
  std::vector<T> vec(size);
  in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));

  return vec;
}

inline std::string format_filename_indices(int num_data, int k, int dim, data_gen_ data_gen_algo)
{
  std::stringstream ss;
  if (data_gen_algo == NORMAL) {
    ss << "NORMAL_indices_naive_" << num_data << "_" << k << "_" << dim << ".bin";
  } else if (data_gen_algo == BLOBS) {
    ss << "BLOBS_indices_naive_" << num_data << "_" << k << "_" << dim << ".bin";
  } else if (data_gen_algo == SIFT || data_gen_algo == SIFT_MMAP) {
    ss << "/home/coder/data/sift/SIFT_indices_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  } else if (data_gen_algo == GIST) {
    ss << "/home/coder/data/gist/GIST_indices_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  } else if (data_gen_algo == OPENAI5M) {
    ss << "OPENAI5M_indices_naive_" << num_data << "_" << k << "_" << dim << ".bin";
  } else if (data_gen_algo == DEEP || DEEP_MMAP) {
    ss << "/home/coder/data/deep_image/DEEP_indices_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  }
  return ss.str();
}

inline std::string format_filename_distances(int num_data, int k, int dim, data_gen_ data_gen_algo)
{
  std::stringstream ss;
  if (data_gen_algo == NORMAL) {
    ss << "NORMAL_distances_naive_" << num_data << "_" << k << "_" << dim << ".bin";
  } else if (data_gen_algo == BLOBS) {
    ss << "BLOBS_distances_naive_" << num_data << "_" << k << "_" << dim << ".bin";
  } else if (data_gen_algo == SIFT || data_gen_algo == SIFT_MMAP) {
    ss << "/home/coder/data/sift/SIFT_distances_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  } else if (data_gen_algo == GIST) {
    ss << "/home/coder/data/gist/GIST_distances_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  } else if (data_gen_algo == OPENAI5M) {
    ss << "OPENAI5M_distances_naive_" << num_data << "_" << k << "_" << dim << ".bin";
  } else if (data_gen_algo == DEEP || DEEP_MMAP) {
    ss << "/home/coder/data/deep_image/DEEP_distances_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  }

  return ss.str();
}

// void load_dataset()

using DataT = float;
using IdxT  = int64_t;

std::unordered_map<data_gen_, std::string> dataset_paths;

void initialize_dataset_map()
{
  // Populate the map with data names and their paths
  dataset_paths[SIFT] = "/home/coder/data/sift/sift_base.fbin";
  dataset_paths[DEEP] = "/home/coder/data/deep_image/deep-image-96-angular.fbin";
  dataset_paths[WIKI] = "/home/coder/data/wiki_all/base.88M.fbin";
}

std::unordered_map<std::string, std::tuple<int64_t, int64_t>> weak_scaling_map;
void initialize_weak_scaling_map_map()
{
  // per gpu rows and cols
  weak_scaling_map["1g"] = std::make_tuple(2500000, 100);
  weak_scaling_map["2g"] = std::make_tuple(5000000, 100);
  weak_scaling_map["3g"] = std::make_tuple(7500000, 100);
  weak_scaling_map["4g"] = std::make_tuple(10000000, 100);
}

int main(int argc, char* argv[])
{
  initialize_dataset_map();
  initialize_weak_scaling_map_map();
  auto data_name_         = static_cast<std::string>(argv[1]);
  auto build_algo_        = static_cast<std::string>(argv[2]);
  auto n_nearest_clusters = std::stoul(argv[3]);
  auto n_clusters         = std::stoul(argv[4]);
  auto do_mmap_flag       = std::stoul(argv[5]);  // 1 for do mmap
  bool check_weak_scale   = false;
  std::string g_per_gpu   = "";
  if (argc >= 7) {
    check_weak_scale = true;
    std::cout << check_weak_scale;
    g_per_gpu = static_cast<std::string>(argv[6]);
  }

  data_gen_ data_name;
  knn_build_algo_ build_algo;
  do_mmap_ do_mmap;
  size_t k = 32;

  if (data_name_ == "sift") {
    data_name = SIFT;
  } else if (data_name_ == "deep") {
    data_name = DEEP;
  } else if (data_name_ == "wiki") {
    data_name = WIKI;
  } else if (data_name_ == "rand") {
    data_name = NORMAL;
  }

  if (build_algo_ == "ivfpq") {
    build_algo = IVF_PQ;
  } else if (build_algo_ == "nnd") {
    build_algo = NN_DESCENT;
  } else {
    return 0;
  }

  if (do_mmap_flag == 1lu) {
    do_mmap = MMAP;
  } else {
    do_mmap = NO_MMAP;
  }

  // set resources and initialize clique

  // preparing statistics analyzer
  // using statistics_adaptor =
  // rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
  // // statistics_adaptor mr{rmm::mr::get_current_device_resource_ref()};
  // rmm::device_async_resource_ref mr_{rmm::mr::get_current_device_resource()};
  // statistics_adaptor mr{mr_};
  // // rmm::mr::set_current_device_resource_ref(mr);

  // using statistics_adaptor =
  // rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>; statistics_adaptor
  // mr{rmm::mr::get_current_device_resource()}; rmm::mr::set_current_device_resource_ref(mr);

  // auto [bytes, allocs] = mr.push_counters();
  // auto tmp = raft::make_device_matrix<DataT>(handle, 10, 4);
  // auto cntr = mr.get_bytes_counter();

  // using statistics_adaptor =
  // rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>; statistics_adaptor
  // mr{rmm::mr::get_current_device_resource()}; rmm::mr::set_current_device_resource(&mr);
  // rmm::device_async_resource mr_{mr};

  // raft::resources handle;

  // auto [bytes, allocs] = mr.push_counters();
  // std::cout << "tmp cntrs here should be 0 " << bytes.value << std::endl;
  // auto tmp2 = raft::make_device_matrix<DataT, int64_t>(handle, 20, 4);
  // auto cntr = mr.get_bytes_counter();
  // std::cout << "cntr value " << cntr.value << " peak " << cntr.peak << " total " << cntr.total <<
  // std::endl;

  raft::resources handle;

  auto clique = raft::resource::get_nccl_clique(handle);

  using statistics_adaptor = rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;
  statistics_adaptor mr{rmm::mr::get_current_device_resource()};
  rmm::mr::set_current_device_resource(&mr);

  auto [bytes, allocs] = mr.push_counters();
  std::cout << "tmp cntrs here should be 0 " << bytes.value << std::endl;
  // auto cntr = mr.get_bytes_counter();
  // std::cout << "[MEM PROFILE] cntr value in bytes " << cntr.value << " peak " << cntr.peak << "
  // total " << cntr.total << std::endl;

  raft::host_matrix_view<DataT, int64_t> database_host_view;
  std::optional<raft::host_matrix<DataT, int64_t>> database_h;
  std::vector<DataT> database_h_vec;
  uint32_t n_rows, n_dim;

  // loading datasets
  if (data_name == NORMAL) {  // for weak scaling check
    auto [row_per_gpu, dim] = weak_scaling_map[g_per_gpu];
    n_rows = row_per_gpu * static_cast<size_t>(clique.num_ranks_) / n_nearest_clusters;
    n_dim  = dim;
    database_h_vec.resize(n_rows * n_dim);

    std::random_device rd;                           // Seed for the random number generator
    std::mt19937 gen(rd());                          // Mersenne Twister engine
    std::uniform_real_distribution<> dis(0.0, 1.0);  // Uniform distribution between 0 and 1

    // Fill the vector with random floats
    for (size_t i = 0; i < database_h_vec.size(); ++i) {
      database_h_vec[i] = dis(gen);
    }

    database_host_view =
      raft::make_host_matrix_view<DataT, int64_t>(database_h_vec.data(), n_rows, n_dim);

  } else {
    std::string input_bin_file = dataset_paths[data_name];
    if (do_mmap == MMAP) {
      std::cout << "MMAP dataset\n";
      DataT* database_h_mmap_ptr;
      int fd = open(input_bin_file.c_str(), O_RDONLY);

      struct stat file_stat;
      fstat(fd, &file_stat);
      size_t file_size    = file_stat.st_size;
      void* mapped_memory = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
      close(fd);

      n_rows = *reinterpret_cast<uint32_t*>(mapped_memory);
      n_dim  = *(reinterpret_cast<uint32_t*>(mapped_memory) + 1);

      database_h_mmap_ptr =
        reinterpret_cast<DataT*>(static_cast<char*>(mapped_memory) + 2 * sizeof(uint32_t));
      database_host_view =
        raft::make_host_matrix_view<DataT, int64_t>(database_h_mmap_ptr, n_rows, n_dim);
    } else {
      std::cout << "NON-MMAP dataset\n";
      std::ifstream file(input_bin_file, std::ios::binary);

      file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
      file.read(reinterpret_cast<char*>(&n_dim), sizeof(n_dim));

      database_h.emplace(raft::make_host_matrix<DataT, int64_t>(n_rows, n_dim));

      file.read(reinterpret_cast<char*>(database_h.value().data_handle()),
                n_rows * n_dim * sizeof(DataT));
      database_host_view = raft::make_host_matrix_view<DataT, int64_t>(
        database_h.value().data_handle(), n_rows, n_dim);
    }
  }

  std::cout << "Number of rows: " << n_rows << std::endl;
  std::cout << "Number of features: " << n_dim << std::endl;

  // batch index params
  cuvs::neighbors::batch_ann::index_params params;
  params.n_clusters         = n_clusters;
  params.n_nearest_clusters = n_nearest_clusters;
  params.metric             = cuvs::distance::DistanceType::L2Expanded;

  if (build_algo == NN_DESCENT) {
    auto nn_descent_params = cuvs::neighbors::batch_ann::graph_build_params::nn_descent_params{};
    nn_descent_params.max_iterations            = 100;
    nn_descent_params.graph_degree              = k;
    nn_descent_params.intermediate_graph_degree = k * 2;
    params.graph_build_params                   = nn_descent_params;
  } else if (build_algo == IVF_PQ) {
    auto ivfq_build_params = cuvs::neighbors::batch_ann::graph_build_params::ivf_pq_params{};
    //    ivfq_build_params.build_params.n_lists = 100;
    ivfq_build_params.build_params.n_lists = std::max(
      5u, static_cast<uint32_t>(n_rows * params.n_nearest_clusters / (5000 * params.n_clusters)));
    params.graph_build_params = ivfq_build_params;
    std::cout << "heuristically good ivfpq n lists: " << ivfq_build_params.build_params.n_lists
              << std::endl;
  }

  auto start = raft::curTimeMillis();
  auto index = cuvs::neighbors::batch_ann::build(
    handle, raft::make_const_mdspan(database_host_view), static_cast<int64_t>(k), params);
  auto end = raft::curTimeMillis();

  auto cntr = mr.get_bytes_counter();
  std::cout << "[MEM PROFILE] cntr value in bytes " << cntr.value << " peak " << cntr.peak
            << " total " << cntr.total << std::endl;

  std::cout << "time to run batch build: " << end - start << std::endl;
  printf("\n\n");

  if (data_name != WIKI && data_name != NORMAL) {
    size_t queries_size = static_cast<size_t>(n_rows) * static_cast<size_t>(k);
    std::vector<IdxT> indices_batch(queries_size);
    raft::copy(indices_batch.data(),
               index.graph().data_handle(),
               queries_size,
               raft::resource::get_cuda_stream(handle));
    std::vector<IdxT> indices_naive =
      read_vector_from_file<IdxT>(format_filename_indices(n_rows, k, n_dim, data_name));
    eval_recall(indices_naive, indices_batch, n_rows, k, 0.01, 2.0, true);
  }

  return 0;
}
