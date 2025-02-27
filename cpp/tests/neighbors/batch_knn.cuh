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
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/nccl_clique.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>
#include <rmm/device_uvector.hpp>

#include "naive_knn.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>

#include <cuvs/neighbors/batch_knn.hpp>
#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace cuvs::neighbors::batch_knn {

enum knn_build_algo { NN_DESCENT, IVF_PQ };

struct BatchKNNInputs {
  knn_build_algo build_algo;
  std::tuple<double, size_t, size_t> recall_cluster_nearestcluster;
  int n_rows;
  int dim;
  int k;
  cuvs::distance::DistanceType metric;
};

inline ::std::ostream& operator<<(::std::ostream& os, const BatchKNNInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.k
     << ", metric=" << static_cast<int>(p.metric)
     << ", clusters=" << std::get<1>(p.recall_cluster_nearestcluster)
     << ", num nearest clusters=" << std::get<2>(p.recall_cluster_nearestcluster) << std::endl;
  return os;
}

template <typename DistanceT, typename DataT, typename IdxT = int64_t>
class BatchKNNTest : public ::testing::TestWithParam<BatchKNNInputs> {
 public:
  BatchKNNTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      clique_(raft::resource::get_nccl_clique(handle_)),
      ps(::testing::TestWithParam<BatchKNNInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void run()
  {
    size_t queries_size = ps.n_rows * ps.k;
    std::vector<IdxT> indices_batch(queries_size);
    std::vector<DistanceT> distances_batch(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

      naive_knn<DistanceT, DataT, IdxT>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        database.data(),
                                        database.data(),
                                        ps.n_rows,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.k,
                                        ps.metric);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      {
        index_params params;  // NEED to change build_algo to be part of params
        if (ps.build_algo == NN_DESCENT) {
          auto nn_descent_params           = graph_build_params::nn_descent_params{};
          nn_descent_params.max_iterations = 100;
          params.graph_build_params        = nn_descent_params;
        } else if (ps.build_algo == IVF_PQ) {
          params.graph_build_params = graph_build_params::ivf_pq_params{};
        }
        params.n_clusters           = std::get<1>(ps.recall_cluster_nearestcluster);
        params.num_nearest_clusters = std::get<2>(ps.recall_cluster_nearestcluster);

        {
          // making dataset
          auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
          raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);

          auto start = raft::curTimeMillis();
          auto index = batch_knn::build(handle_,
                                        raft::make_const_mdspan(database_host.view()),
                                        static_cast<int64_t>(ps.k),
                                        params);
          auto end   = raft::curTimeMillis();
          std::cout << "time to run batch build: " << end - start << std::endl;

          raft::copy(indices_batch.data(), index.graph().data_handle(), queries_size, stream_);
          if (index.distances().has_value()) {
            raft::copy(distances_batch.data(),
                       index.distances().value().data_handle(),
                       queries_size,
                       stream_);
          }
        }
        raft::resource::sync_stream(handle_);
      }
      double min_recall = std::get<0>(ps.recall_cluster_nearestcluster);

      std::cout << "eval recall\n";
      EXPECT_TRUE(
        eval_recall(indices_naive, indices_batch, ps.n_rows, ps.k, 0.01, min_recall, true));
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
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
  BatchKNNInputs ps;
  rmm::device_uvector<DataT> database;
};

const std::vector<BatchKNNInputs> inputsBatch = raft::util::itertools::product<BatchKNNInputs>(
  // {NN_DESCENT, IVF_PQ},
  {IVF_PQ},
  {std::make_tuple(0.85, 4lu, 2lu),
   std::make_tuple(0.95, 10lu, 5lu),
   std::make_tuple(0.95, 11lu, 7lu)},  // min_recall, n_clusters, num_nearest_cluster
  {10000},                             // n_rows
  {192, 256},                          // dim
  {32, 64},                            // graph_degree
  {cuvs::distance::DistanceType::L2Expanded});

}  // namespace cuvs::neighbors::batch_knn
