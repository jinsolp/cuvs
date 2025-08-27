/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "../../sample_filter.cuh"

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra::detail {

struct none_filter_args_t {};
using bitset_filter_args_t = cuvs::core::bitset_view<uint32_t, int64_t>;

struct cagra_filter_dev {
  filtering::FilterType tag_;
  uint32_t offset;

  union cagra_filter_dev_args_variant {
    none_filter_args_t none_filter_args;
    bitset_filter_args_t bitset_filter_args;

    _RAFT_HOST_DEVICE explicit cagra_filter_dev_args_variant(const none_filter_args_t& args)
      : none_filter_args(args)
    {
    }

    _RAFT_HOST_DEVICE explicit cagra_filter_dev_args_variant(const bitset_filter_args_t& args)
      : bitset_filter_args(args)
    {
    }
  } args_;

  _RAFT_HOST_DEVICE cagra_filter_dev(none_filter_args_t args = {})
    : tag_(filtering::FilterType::None), args_(args) {};

  _RAFT_HOST_DEVICE cagra_filter_dev(const bitset_filter_args_t& args)
    : tag_(filtering::FilterType::Bitset), args_(args) {};

  constexpr __forceinline__ _RAFT_HOST_DEVICE bool operator()(const uint32_t query_id,
                                                              const uint32_t sample_id)
  {
    switch (tag_) {
      case filtering::FilterType::None:
        return filtering::none_sample_filter{}(query_id + offset, sample_id);
      case filtering::FilterType::Bitset: {
        return filtering::bitset_filter<uint32_t, int64_t>(args_.bitset_filter_args)(
          query_id + offset, sample_id);
      }
      default: return true;
    }
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterWithQueryIdOffset {
  const uint32_t offset;
  CagraSampleFilterT filter;

  CagraSampleFilterWithQueryIdOffset(const uint32_t offset, const CagraSampleFilterT filter)
    : offset(offset), filter(filter)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id + offset, sample_id);
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterT_Selector {
  using type = CagraSampleFilterWithQueryIdOffset<CagraSampleFilterT>;
};
template <>
struct CagraSampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter> {
  using type = cuvs::neighbors::filtering::none_sample_filter;
};

// A helper function to set a query id offset
template <class CagraSampleFilterT>
inline typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type set_offset(
  CagraSampleFilterT filter, const uint32_t offset)
{
  typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type new_filter(offset, filter);
  return new_filter;
}
template <>
inline typename CagraSampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter>::type
set_offset<cuvs::neighbors::filtering::none_sample_filter>(
  cuvs::neighbors::filtering::none_sample_filter filter, const uint32_t)
{
  return filter;
}
}  // namespace cuvs::neighbors::cagra::detail
