/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "batch_knn.cuh"
#include "cuvs/neighbors/common.hpp"
//  #include <cuvs/neighbors/batch_knn.hpp>
//  #include "cuvs/neighbors/ivf_pq.hpp"

namespace cuvs::neighbors::batch_knn {
// using namespace cuvs::neighbors;

#define CUVS_INST_BATCH_KNN_IVF_PQ(T, IdxT)                                                    \
  batch_knn::index<IdxT, T> build(const raft::resources& handle,                               \
                                  raft::host_matrix_view<const T, int64_t, row_major> dataset, \
                                  int64_t k,                                                   \
                                  const index_params& params)                                  \
  {                                                                                            \
    return batch_knn::detail::build<T, IdxT>(handle, dataset, k, params);                      \
  }

CUVS_INST_BATCH_KNN_IVF_PQ(float, int64_t);

#undef CUVS_INST_BATCH_KNN_IVF_PQ

}  // namespace cuvs::neighbors::batch_knn
