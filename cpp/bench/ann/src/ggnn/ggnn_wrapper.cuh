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

#include "../common/ann_types.hpp"
#include "../common/util.hpp"

// #include <raft/util/cudart_utils.hpp>

#include <ggnn/cuda_knn_ggnn_gpu_instance.cuh>

#include <memory>
#include <stdexcept>

namespace cuvs::bench {

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
class ggnn_impl;

template <typename T>
class ggnn : public algo<T>, public algo_gpu {
 public:
  struct build_param {
    int k_build{24};       // KBuild
    int segment_size{32};  // S
    int num_layers{4};     // L
    float tau{0.5};
    int refine_iterations{2};
    int k;  // GGNN requires to know k during building
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    float tau;
    int block_dim{32};
    int max_iterations{400};
    int cache_size{512};
    int sorted_size{256};
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
  };

  ggnn(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override { impl_->build(dataset, nrow); }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    impl_->set_search_param(param, filter_bitset);
  }
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    impl_->search(queries, batch_size, k, neighbors, distances);
  }
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return dynamic_cast<algo_gpu*>(impl_.get())->get_sync_stream();
  }

  void save(const std::string& file) const override { impl_->save(file); }
  void load(const std::string& file) override { impl_->load(file); }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<ggnn<T>>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return impl_->get_preference();
  }

  void set_search_dataset(const T* dataset, size_t nrow) override
  {
    impl_->set_search_dataset(dataset, nrow);
  };

 private:
  std::shared_ptr<algo<T>> impl_;
};

template <typename T>
ggnn<T>::ggnn(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  // ggnn/src/sift1m.cu
  if (metric == Metric::kEuclidean && dim == 128 && param.k_build == 24 && param.k == 10 &&
      param.segment_size == 32) {
    impl_ = std::make_shared<ggnn_impl<T, Euclidean, 128, 24, 10, 32>>(metric, dim, param);
  }
  // ggnn/src/deep1b_multi_gpu.cu, and adapt it deep1B
  else if (metric == Metric::kEuclidean && dim == 96 && param.k_build == 24 && param.k == 10 &&
           param.segment_size == 32) {
    impl_ = std::make_shared<ggnn_impl<T, Euclidean, 96, 24, 10, 32>>(metric, dim, param);
  } else if (metric == Metric::kInnerProduct && dim == 96 && param.k_build == 24 && param.k == 10 &&
             param.segment_size == 32) {
    impl_ = std::make_shared<ggnn_impl<T, Cosine, 96, 24, 10, 32>>(metric, dim, param);
  } else if (metric == Metric::kInnerProduct && dim == 96 && param.k_build == 96 && param.k == 10 &&
             param.segment_size == 64) {
    impl_ = std::make_shared<ggnn_impl<T, Cosine, 96, 96, 10, 64>>(metric, dim, param);
  }
  // ggnn/src/glove200.cu, adapt it to glove100
  else if (metric == Metric::kInnerProduct && dim == 100 && param.k_build == 96 && param.k == 10 &&
           param.segment_size == 64) {
    impl_ = std::make_shared<ggnn_impl<T, Cosine, 100, 96, 10, 64>>(metric, dim, param);
  } else {
    throw std::runtime_error(
      "ggnn: not supported combination of metric, dim and build param; "
      "see Ggnn's constructor in ggnn_wrapper.cuh for available combinations");
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
class ggnn_impl : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  ggnn_impl(Metric metric, int dim, const typename ggnn<T>::build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override { return stream_; }

  void save(const std::string& file) const override;
  void load(const std::string& file) override;
  std::unique_ptr<algo<T>> copy() override
  {
    auto r = std::make_unique<ggnn_impl<T, measure, D, KBuild, KQuery, S>>(*this);
    // set the thread-local stream to the copied handle.
    r->stream_ = cuvs::bench::get_stream_from_global_pool();
    return r;
  };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kDevice;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }

  void set_search_dataset(const T* dataset, size_t nrow) override;

 private:
  using algo<T>::metric_;
  using algo<T>::dim_;

  using ggnngpu_instance = GGNNGPUInstance<measure,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           T /* BaseT */,
                                           size_t /* BAddrT */,
                                           D,
                                           KBuild,
                                           KBuild / 2 /* KF */,
                                           KQuery,
                                           S>;
  std::shared_ptr<ggnngpu_instance> ggnn_;
  typename ggnn<T>::build_param build_param_;
  typename ggnn<T>::search_param search_param_;
  cudaStream_t stream_;
  const T* base_dataset_                 = nullptr;
  size_t base_n_rows_                    = 0;
  std::optional<std::string> graph_file_ = std::nullopt;

  void load_impl()
  {
    if (base_dataset_ == nullptr) { return; }
    if (base_n_rows_ == 0) { return; }
    int device;
    cudaGetDevice(&device);
    ggnn_ = std::make_shared<ggnngpu_instance>(
      device, base_n_rows_, build_param_.num_layers, true, build_param_.tau);
    ggnn_->set_base_data(base_dataset_);
    ggnn_->set_stream(get_sync_stream());
    if (graph_file_.has_value()) {
      auto& ggnn_host   = ggnn_->ggnn_cpu_buffers.at(0);
      auto& ggnn_device = ggnn_->ggnn_shards.at(0);
      ggnn_->set_stream(get_sync_stream());

      ggnn_host.load(graph_file_.value());
      ggnn_host.uploadAsync(ggnn_device);
      cudaStreamSynchronize(ggnn_device.stream);
    }
  }
};

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
ggnn_impl<T, measure, D, KBuild, KQuery, S>::ggnn_impl(Metric metric,
                                                       int dim,
                                                       const typename ggnn<T>::build_param& param)
  : algo<T>(metric, dim), build_param_(param), stream_(cuvs::bench::get_stream_from_global_pool())
{
  if (metric_ == Metric::kInnerProduct) {
    if (measure != Cosine) { throw std::runtime_error("mis-matched metric"); }
  } else if (metric_ == Metric::kEuclidean) {
    if (measure != Euclidean) { throw std::runtime_error("mis-matched metric"); }
  } else {
    throw std::runtime_error(
      "ggnn supports only metric type of InnerProduct, Cosine and Euclidean");
  }

  if (dim != D) { throw std::runtime_error("mis-matched dim"); }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::build(const T* dataset, size_t nrow)
{
  base_dataset_ = dataset;
  base_n_rows_  = nrow;
  graph_file_   = std::nullopt;
  load_impl();
  ggnn_->build(0);
  for (int i = 0; i < build_param_.refine_iterations; ++i) {
    ggnn_->refine();
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::set_search_dataset(const T* dataset, size_t nrow)
{
  if (base_dataset_ != dataset || base_n_rows_ != nrow) {
    base_dataset_ = dataset;
    base_n_rows_  = nrow;
    load_impl();
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::set_search_param(const search_param_base& param,
                                                                   const void* filter_bitset)
{
  if (filter_bitset != nullptr) { throw std::runtime_error("Filtering is not supported yet."); }
  search_param_ = dynamic_cast<const typename ggnn<T>::search_param&>(param);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(sizeof(size_t) == sizeof(int64_t), "sizes of size_t and GGNN's KeyT are different");
  if (k != KQuery) {
    throw std::runtime_error(
      "k = " + std::to_string(k) +
      ", but this GGNN instance only supports k = " + std::to_string(KQuery));
  }

  ggnn_->set_stream(get_sync_stream());
  cudaMemcpyToSymbol(c_tau_query, &search_param_.tau, sizeof(float));

  const int block_dim      = search_param_.block_dim;
  const int max_iterations = search_param_.max_iterations;
  const int cache_size     = search_param_.cache_size;
  const int sorted_size    = search_param_.sorted_size;
  // default value
  if (block_dim == 32 && max_iterations == 400 && cache_size == 512 && sorted_size == 256) {
    ggnn_->template queryLayer<32, 400, 512, 256, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/sift1m.cu
  else if (block_dim == 32 && max_iterations == 200 && cache_size == 256 && sorted_size == 64) {
    ggnn_->template queryLayer<32, 200, 256, 64, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/sift1m.cu
  else if (block_dim == 32 && max_iterations == 400 && cache_size == 448 && sorted_size == 64) {
    ggnn_->template queryLayer<32, 400, 448, 64, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/glove200.cu
  else if (block_dim == 128 && max_iterations == 2000 && cache_size == 2048 && sorted_size == 32) {
    ggnn_->template queryLayer<128, 2000, 2048, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // for glove100
  else if (block_dim == 64 && max_iterations == 400 && cache_size == 512 && sorted_size == 32) {
    ggnn_->template queryLayer<64, 400, 512, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else if (block_dim == 128 && max_iterations == 2000 && cache_size == 1024 &&
             sorted_size == 32) {
    ggnn_->template queryLayer<128, 2000, 1024, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else {
    throw std::runtime_error("ggnn: not supported search param");
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::save(const std::string& file) const
{
  auto& ggnn_host   = ggnn_->ggnn_cpu_buffers.at(0);
  auto& ggnn_device = ggnn_->ggnn_shards.at(0);
  ggnn_->set_stream(get_sync_stream());

  ggnn_host.downloadAsync(ggnn_device);
  cudaStreamSynchronize(ggnn_device.stream);
  ggnn_host.store(file);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::load(const std::string& file)
{
  if (!graph_file_.has_value() || graph_file_.value() != file) {
    graph_file_ = file;
    load_impl();
  }
}

}  // namespace cuvs::bench
