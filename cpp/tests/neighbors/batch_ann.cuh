/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <cstdint>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/nccl_clique.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/make_blobs.cuh>
#include <sys/mman.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <rapids_logger/logger.hpp>
#include <rmm/device_uvector.hpp>

#include "naive_knn.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/batch_ann.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace cuvs::neighbors::batch_ann {

enum knn_build_algo { NN_DESCENT, IVF_PQ };
enum data_gen { NORMAL, BLOBS, SIFT, GIST, OPENAI5M, DEEP, SIFT_MMAP };

struct BatchANNInputs {
  data_gen data_gen_algo;
  knn_build_algo build_algo;
  std::tuple<double, size_t, size_t> recall_cluster_nearestcluster;
  int n_rows;
  int dim;
  int k;
  cuvs::distance::DistanceType metric;
};

inline ::std::ostream& operator<<(::std::ostream& os, const BatchANNInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.k
     << ", metric=" << static_cast<int>(p.metric)
     << ", clusters=" << std::get<1>(p.recall_cluster_nearestcluster)
     << ", num nearest clusters=" << std::get<2>(p.recall_cluster_nearestcluster) << std::endl;
  return os;
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

inline std::string format_filename_indices(int num_data, int k, int dim, data_gen data_gen_algo)
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
  } else if (data_gen_algo == DEEP) {
    ss << "/home/coder/data/deep_image/DEEP_indices_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  }
  return ss.str();
}

inline std::string format_filename_distances(int num_data, int k, int dim, data_gen data_gen_algo)
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
  } else if (data_gen_algo == DEEP) {
    ss << "/home/coder/data/deep_image/DEEP_distances_naive_" << num_data << "_" << k << "_" << dim
       << ".bin";
  }

  return ss.str();
}

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
class BatchANNTest : public ::testing::TestWithParam<BatchANNInputs> {
 public:
  BatchANNTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      clique_(raft::resource::get_nccl_clique(handle_)),
      ps(::testing::TestWithParam<BatchANNInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void run()
  {
    index_params params;
    params.n_clusters         = std::get<1>(ps.recall_cluster_nearestcluster);
    params.n_nearest_clusters = std::get<2>(ps.recall_cluster_nearestcluster);
    params.metric             = ps.metric;

    if (ps.build_algo == NN_DESCENT) {
      auto nn_descent_params                      = graph_build_params::nn_descent_params{};
      nn_descent_params.max_iterations            = 1000;
      nn_descent_params.graph_degree              = ps.k;
      nn_descent_params.intermediate_graph_degree = ps.k * 2;
      params.graph_build_params                   = nn_descent_params;
    } else if (ps.build_algo == IVF_PQ) {
      auto ivfq_build_params = graph_build_params::ivf_pq_params{};
      //    ivfq_build_params.build_params.n_lists = 100;
      ivfq_build_params.build_params.n_lists = std::max(
        5u,
        static_cast<uint32_t>(ps.n_rows * params.n_nearest_clusters / (5000 * params.n_clusters)));
      params.graph_build_params = ivfq_build_params;
      std::cout << "heuristically good ivfpq n lists: " << ivfq_build_params.build_params.n_lists
                << std::endl;
    }

    size_t queries_size = ps.n_rows * ps.k;
    std::cout << "queries size: " << queries_size << std::endl;

    std::vector<IdxT> indices_batch(queries_size);
    std::vector<DistanceT> distances_batch(queries_size);
    // std::vector<DistanceT> distances_naive(queries_size);
    // std::vector<IdxT> indices_naive(queries_size);
    std::vector<IdxT> indices_naive = read_vector_from_file<IdxT>(
      format_filename_indices(ps.n_rows, ps.k, ps.dim, ps.data_gen_algo));
    std::vector<DistanceT> distances_naive = read_vector_from_file<DistanceT>(
      format_filename_distances(ps.n_rows, ps.k, ps.dim, ps.data_gen_algo));

    // {
    //   rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
    //   rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    //   std::cout << "rows: " << ps.n_rows << ", dims: " << ps.dim << std::endl;
    //   auto start = raft::curTimeMillis();
    //   naive_knn<DistanceT, DataT, IdxT>(handle_,
    //                                     distances_naive_dev.data(),
    //                                     indices_naive_dev.data(),
    //                                     database.data(),
    //                                     database.data(),
    //                                     ps.n_rows,
    //                                     ps.n_rows,
    //                                     ps.dim,
    //                                     ps.k,
    //                                     ps.metric);
    //   auto end = raft::curTimeMillis();
    //   std::cout << "time to run naive build: " << end - start << std::endl;
    //   raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
    //   raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size,
    //   stream_);

    //   write_vector_to_file(indices_naive, format_filename_indices(ps.n_rows, ps.k, ps.dim,
    //   ps.data_gen_algo)); write_vector_to_file(distances_naive,
    //   format_filename_distances(ps.n_rows, ps.k, ps.dim, ps.data_gen_algo));
    //   raft::resource::sync_stream(handle_);
    //   std::cout << "done writing indices and distances\n";
    // }
    std::cout << "done reading naive indices and distances\n";
    {
      {
        {
          // making dataset
          // auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          // raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
          uint32_t start, end;

          if (ps.data_gen_algo == SIFT_MMAP) {
            std::cout << "mmap batch build\n";
            auto database_host_view =
              raft::make_host_matrix_view<DataT, int64_t>(database_h_mmap_ptr, ps.n_rows, ps.dim);
            start      = raft::curTimeMillis();
            auto index = batch_ann::build(handle_,
                                          raft::make_const_mdspan(database_host_view),
                                          static_cast<int64_t>(ps.k),
                                          params);
            end        = raft::curTimeMillis();
            raft::copy(indices_batch.data(), index.graph().data_handle(), queries_size, stream_);
            if (index.distances().has_value()) {
              raft::copy(distances_batch.data(),
                         index.distances().value().data_handle(),
                         queries_size,
                         stream_);
            }
          } else {
            std::cout << "NON-mmap batch build\n";
            auto database_host_view = raft::make_host_matrix_view<DataT, int64_t>(
              database_h.value().data_handle(), ps.n_rows, ps.dim);

            start      = raft::curTimeMillis();
            auto index = batch_ann::build(handle_,
                                          // raft::make_const_mdspan(database_host.view()),
                                          raft::make_const_mdspan(database_host_view),
                                          static_cast<int64_t>(ps.k),
                                          params);

            end = raft::curTimeMillis();

            raft::copy(indices_batch.data(), index.graph().data_handle(), queries_size, stream_);
            if (index.distances().has_value()) {
              raft::copy(distances_batch.data(),
                         index.distances().value().data_handle(),
                         queries_size,
                         stream_);
            }
          }
          std::cout << "time to run batch build: " << end - start << std::endl;
        }
        raft::resource::sync_stream(handle_);
      }
      double min_recall = std::get<0>(ps.recall_cluster_nearestcluster);

      EXPECT_TRUE(
        eval_recall(indices_naive, indices_batch, ps.n_rows, ps.k, 0.01, min_recall, true));
    }
  }

  void SetUp() override
  {
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      if (ps.data_gen_algo == NORMAL) {
        database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
        std::cout << "making normal data\n";
        raft::random::normal(
          handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
        database_h.emplace(raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim));
        raft::copy(database_h.value().data_handle(),
                   database.data(),
                   ps.n_rows * ps.dim,
                   raft::resource::get_cuda_stream(handle_));
      } else if (ps.data_gen_algo == BLOBS) {
        database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
        std::cout << "making blobs data\n";
        auto database_view =
          raft::make_device_matrix_view<float, IdxT>(database.data(), ps.n_rows, ps.dim);
        auto labels = raft::make_device_vector<IdxT, IdxT>(handle_, ps.n_rows);
        raft::random::make_blobs(handle_, database_view, labels.view());
        database_h.emplace(raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim));
        raft::copy(database_h.value().data_handle(),
                   database.data(),
                   ps.n_rows * ps.dim,
                   raft::resource::get_cuda_stream(handle_));
      } else if (ps.data_gen_algo == SIFT) {
        // std::string input_bin_file = "/home/coder/data/sift/sift-128-euclidean.fbin";
        std::string input_bin_file = "/home/coder/data/sift/sift_base.fbin";

        std::ifstream file(input_bin_file, std::ios::binary);

        file.seekg(0, std::ios::end);              // Move to the end of the file
        std::streamsize file_size = file.tellg();  // Get the size
        file.seekg(0, std::ios::beg);              // Move back to the beginning
        std::size_t num_floats =
          file_size / sizeof(float);  // Assuming the file contains only floats
        std::cout << "num floats: " << num_floats << std::endl;

        // Read the header (n_samples and n_features)
        uint32_t n_samples, n_features;
        file.read(reinterpret_cast<char*>(&n_samples), sizeof(n_samples));
        file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));
        std::cout << "SIFT Number of samples: " << n_samples << std::endl;
        std::cout << "SIFT Number of features: " << n_features << std::endl;

        database_h.emplace(raft::make_host_matrix<DataT, int64_t>(
          n_samples, n_features));  // resize(n_samples * n_features);
        file.read(reinterpret_cast<char*>(database_h.value().data_handle()),
                  n_samples * n_features * sizeof(float));

        database.resize(n_samples * n_features, stream_);
        raft::print_host_vector("data on host", database_h.value().data_handle(), 10, std::cout);
        raft::print_host_vector("data on host last parts",
                                database_h.value().data_handle() + n_samples * n_features - 10,
                                10,
                                std::cout);
        ps.n_rows = n_samples;
        ps.dim    = n_features;
        raft::copy(database.data(),
                   database_h.value().data_handle(),
                   ps.n_rows * ps.dim,
                   raft::resource::get_cuda_stream(handle_));

      } else if (ps.data_gen_algo == GIST) {
        // database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
        std::string input_bin_file = "/home/coder/data/gist/gist-960-euclidean.fbin";
        std::ifstream file(input_bin_file, std::ios::binary);

        file.seekg(0, std::ios::end);              // Move to the end of the file
        std::streamsize file_size = file.tellg();  // Get the size
        file.seekg(0, std::ios::beg);              // Move back to the beginning
        std::size_t num_floats =
          file_size / sizeof(float);  // Assuming the file contains only floats
        std::cout << "num floats: " << num_floats << std::endl;

        // Read the header (n_samples and n_features)
        uint32_t n_samples, n_features;
        file.read(reinterpret_cast<char*>(&n_samples), sizeof(n_samples));
        file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));
        std::cout << "GIST Number of samples: " << n_samples << std::endl;
        std::cout << "GIST Number of features: " << n_features << std::endl;

        database_h.emplace(raft::make_host_matrix<DataT, int64_t>(
          n_samples, n_features));  // resize(n_samples * n_features);
        file.read(reinterpret_cast<char*>(database_h.value().data_handle()),
                  n_samples * n_features * sizeof(float));
        // std::cout << "size of database is " << database_h.value().size() << std::endl;

        database.resize(n_samples * n_features, stream_);
        raft::print_host_vector("data on host", database_h.value().data_handle(), 10, std::cout);
        raft::print_host_vector("data on host last parts",
                                database_h.value().data_handle() + n_samples * n_features - 10,
                                10,
                                std::cout);
        ps.n_rows = n_samples;
        ps.dim    = n_features;
        raft::copy(database.data(),
                   database_h.value().data_handle(),
                   ps.n_rows * ps.dim,
                   raft::resource::get_cuda_stream(handle_));

      } else if (ps.data_gen_algo == OPENAI5M) {
        const std::string filename = "/home/coder/data/openai_5M/base.5M.fbin";
        std::ifstream file(filename, std::ios::binary);

        file.seekg(0, std::ios::end);              // Move to the end of the file
        std::streamsize file_size = file.tellg();  // Get the size
        file.seekg(0, std::ios::beg);              // Move back to the beginning
        std::size_t num_floats =
          file_size / sizeof(float);  // Assuming the file contains only floats
        std::cout << "num floats: " << num_floats << std::endl;

        uint32_t n_samples  = 0;
        uint32_t n_features = 0;
        // Read the first 8 bytes (4 bytes for n_samples, 4 bytes for n_features)
        file.read(reinterpret_cast<char*>(&n_samples), sizeof(n_samples));
        file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));
        std::cout << "OPENAI Number of samples: " << n_samples << std::endl;
        std::cout << "OPENAI Number of features: " << n_features << std::endl;

        database_h.emplace(raft::make_host_matrix<DataT, int64_t>(
          n_samples, n_features));  // resize(n_samples * n_features);
        file.read(reinterpret_cast<char*>(database_h.value().data_handle()),
                  n_samples * n_features * sizeof(float));
        // std::cout << "size of database is " << database_h.value().size() << std::endl;

        database.resize(n_samples * n_features, stream_);
        raft::print_host_vector("data on host", database_h.value().data_handle(), 10, std::cout);
        raft::print_host_vector("data on host last parts",
                                database_h.value().data_handle() + n_samples * n_features - 10,
                                10,
                                std::cout);
        ps.n_rows = n_samples;
        ps.dim    = n_features;

        raft::copy(database.data(),
                   database_h.value().data_handle(),
                   ps.n_rows * ps.dim,
                   raft::resource::get_cuda_stream(handle_));
      } else if (ps.data_gen_algo == DEEP) {
        const std::string input_bin_file = "/home/coder/data/deep_image/deep-image-96-angular.fbin";
        std::ifstream file(input_bin_file, std::ios::binary);

        file.seekg(0, std::ios::end);              // Move to the end of the file
        std::streamsize file_size = file.tellg();  // Get the size
        file.seekg(0, std::ios::beg);              // Move back to the beginning
        std::cout << "file size: " << file_size << std::endl;
        std::size_t num_floats =
          file_size / sizeof(float);  // Assuming the file contains only floats
        std::cout << "num floats: " << num_floats << std::endl;

        uint32_t n_samples  = 0;
        uint32_t n_features = 0;
        // Read the first 8 bytes (4 bytes for n_samples, 4 bytes for n_features)
        file.read(reinterpret_cast<char*>(&n_samples), sizeof(n_samples));
        file.read(reinterpret_cast<char*>(&n_features), sizeof(n_features));
        std::cout << "DEEP Number of samples: " << n_samples << std::endl;
        std::cout << "DEEP Number of features: " << n_features << std::endl;

        database_h.emplace(raft::make_host_matrix<DataT, int64_t>(
          n_samples, n_features));  // resize(n_samples * n_features);
        file.read(reinterpret_cast<char*>(database_h.value().data_handle()),
                  n_samples * n_features * sizeof(float));
        // std::cout << "size of database is " << database_h.value().size() << std::endl;

        // database.resize(n_samples * n_features, stream_);
        raft::print_host_vector("data on host", database_h.value().data_handle(), 10, std::cout);
        raft::print_host_vector("data on host last parts",
                                database_h.value().data_handle() + n_samples * n_features - 10,
                                10,
                                std::cout);
        ps.n_rows = n_samples;
        ps.dim    = n_features;

        // raft::copy(database.data(), database_h.value().data_handle(), ps.n_rows * ps.dim,
        // raft::resource::get_cuda_stream(handle_));
      } else if (ps.data_gen_algo == SIFT_MMAP) {
        std::string input_bin_file = "/home/coder/data/sift/sift_base.fbin";
        // Open the file
        int fd = open(input_bin_file.c_str(), O_RDONLY);

        // Get the size of the file
        struct stat file_stat;
        fstat(fd, &file_stat);

        size_t file_size       = file_stat.st_size;
        std::size_t num_floats = file_size / sizeof(float);
        std::cout << "num floats: " << num_floats << std::endl;

        // Memory-map the file
        void* mapped_memory = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);

        // Close the file descriptor, as we no longer need it
        close(fd);

        // Access the data
        uint32_t n_samples  = *reinterpret_cast<uint32_t*>(mapped_memory);
        uint32_t n_features = *(reinterpret_cast<uint32_t*>(mapped_memory) + 1);

        std::cout << "SIFT MMAP Number of samples: " << n_samples << std::endl;
        std::cout << "SIFT MMAP Number of features: " << n_features << std::endl;

        // If you want to access the rest of the data (as floats)
        database_h_mmap_ptr =
          reinterpret_cast<DataT*>(static_cast<char*>(mapped_memory) + 2 * sizeof(uint32_t));
        ps.n_rows = n_samples;
        ps.dim    = n_features;
      }
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
    std::cout << "data is prepared\n";
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  raft::comms::nccl_clique clique_;
  BatchANNInputs ps;
  rmm::device_uvector<DataT> database;
  // std::vector<DataT> database_h;
  std::optional<raft::host_matrix<DataT, int64_t>> database_h;
  DataT* database_h_mmap_ptr;
};

const std::vector<BatchANNInputs> inputsBatch = raft::util::itertools::product<BatchANNInputs>(
  //    {SIFT, GIST, DEEP},
  {DEEP},
  //    {NN_DESCENT, IVF_PQ},
  {IVF_PQ},
  // {std::make_tuple(1.0, 1lu, 1lu)},  // min_recall, n_clusters, num_nearest_cluster
  {
    std::make_tuple(1.0, 10lu, 2lu)
    // ,
    // std::make_tuple(1.0, 10lu, 2lu),
    // std::make_tuple(1.0, 1lu, 1lu)
  },        // min_recall, n_clusters, num_nearest_cluster
  {10000},  // n_rows
  {128},    // dim
  {32},     // graph_degree
  {cuvs::distance::DistanceType::L2Expanded});

}  // namespace cuvs::neighbors::batch_ann
