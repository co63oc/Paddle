// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {
using FeedType =
    paddle::variant<phi::DenseTensor, phi::Strings, phi::SparseCooTensor>;
using FetchType = paddle::variant<phi::DenseTensor,
                                  phi::TensorArray,
                                  phi::Vocab,
                                  phi::SparseCooTensor>;

template <>
struct PhiVectorType<FeedType> {
  const char *type_name = "PhiVectorFeedType";
};

template <>
struct PhiVectorType<FetchType> {
  const char *type_name = "PhiVectorFetchType";
};

using FeedList = PhiVector<FeedType>;
using FetchList = PhiVector<FetchType>;
using FetchUnmergedList = std::vector<std::vector<FetchType>>;

inline bool data_is_lod_tensor(const FetchType &data) {
  if (data.type() == typeid(phi::DenseTensor)) {
    return true;
  }
  return false;
}

inline bool data_is_lod_tensor_array(const FetchType &data) {
  if (data.type() == typeid(phi::TensorArray)) {
    return true;
  }
  return false;
}

inline bool data_is_string_tensor(const FeedType &data) {
  if (data.type() == typeid(Strings)) {
    return true;
  }
  return false;
}

inline bool data_is_sparse_coo_tensor(const FetchType &data) {
  if (data.type() == typeid(phi::SparseCooTensor)) {
    return true;
  }
  return false;
}

static const char kFeedOpType[] = "feed";
static const char kFetchOpType[] = "fetch";

}  // namespace phi
