/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/framework/tensor_util.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace framework {

// Split phi::DenseTensor and copy to each place specified in places.
TEST_API std::vector<phi::DenseTensor> SplitLoDTensor(
    const phi::DenseTensor& src, const std::vector<phi::Place> places);

TEST_API void MergeLoDTensor(
    phi::DenseTensor* target,
    const std::vector<const phi::DenseTensor*>& lod_tensors,
    phi::Place dst_place);

/*
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
using LoD = std::vector<phi::Vector<size_t>>;

std::string LoDToString(const LoD& lod);

LoD SliceInLevel(const LoD& in,
                 size_t level,
                 size_t elem_begin,
                 size_t elem_end);
/*
 * Transform an LoD from relative offsets to absolute offsets.
 */
TEST_API LoD ToAbsOffset(const LoD& in);

TEST_API bool operator==(const LoD& a, const LoD& b);

/*
 * Check whether this lod's format is valid.
 *
 * ATTENTION:
 *   - Empty lod is treated as valid.
 *
 * It will check two things:
 *
 *  1. all the offsets in a level should be non-descending.
 *  2. there should be more than 2 offsets existing in each level.
 *  3. the higher level's last offset should equals the lower level's size-1.
 *  4. the first offset(the begin offset) of each level should be 0.
 *  5. the lowest level's last offset should equals `tensor_height` if
 * tensor_height>0.
 */

TEST_API bool CheckLoD(const LoD& in, int tensor_height = -1);
/*
 * Check whether this absolute lod's format is valid.
 *
 * ATTENTION:
 *   - Empty lod is treated as valid.
 *
 * It will check two things:
 *  1. all the offsets in a level should be ascending(no same items allowed).
 *  2. there should be more than 2 offsets existing in each level.
 *  3. the first offset of each level should be 0, and the last should be the
 *     same(the height of underlying tensor) or `tensor_height` if
 *     tensor_height>0.
 */
TEST_API bool CheckAbsLoD(const LoD& in, int tensor_height = -1);

/*
 * Expand the `source` to fit the LoD of `lod`. For example, a `source`
 * phi::DenseTensor is
 *  - LoD: [0, 2]
 *  - tensor: [a0, a1]
 * a `lod` is
 *  - LoD: [0 3 5]
 * returns a new phi::DenseTensor
 *  - [a0 a0 a0 a1 a1]
 */
template <typename T>
phi::DenseTensor LodExpand(const phi::DenseTensor& source,
                           const LoD& lod,
                           size_t level,
                           const phi::Place& place) {
  LoD abs_lod = ToAbsOffset(lod);
  const auto& lod_level = lod[level];
  size_t num_instances = source.dims()[0];

  // new tensor
  phi::DenseTensor tensor;
  tensor.set_lod(lod);
  auto dims = source.dims();
  dims[0] = lod_level.back();
  tensor.Resize(dims);
  tensor.mutable_data<T>(place);

  PADDLE_ENFORCE_EQ(
      num_instances,
      lod_level.size() - 1,
      common::errors::InvalidArgument(
          "The input phi::DenseTensor instance number should be equal to the "
          "LoD "
          "level size minus 1."
          "The input instance number is %zu, LoD level size is %zu.",
          num_instances,
          lod_level.size()));
  for (size_t ins = 0; ins < num_instances; ins++) {
    for (size_t elem = lod_level[ins]; elem < lod_level[ins + 1]; elem++) {
      auto slice = tensor.Slice(elem, elem + 1);
      TensorCopy(source.Slice(ins, ins + 1),
                 phi::CPUPlace(),
                 phi::CPUContext(),
                 &slice);
    }
  }
  return tensor;
}

// Get the absolute offset of a lod[start_level][start_idx:end_idx] and
// relative length of details for every levels(i.e., [start_level: ]).
//
// For example,
//   lod = [[0, 3, 4, 8], [0, 9, 10, 11, 13, 17, 19, 22, 24]]
//   start_level = 0
//   start_idx = 1
//   end_idx = 3
//
// Returns:
//  LoD = [[1, 4], [2, 4, 2, 3, 2]]
//  pair<size_t, size_t> = {11, 24}
TEST_API std::pair<LoD, std::pair<size_t, size_t>> GetSubLoDAndAbsoluteOffset(
    const LoD& lod, size_t start_idx, size_t end_idx, size_t start_level);

/*
 * Serialize/Deserialize phi::DenseTensor to std::ostream
 * You can pass ofstream or ostringstream to serialize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os,
                       const phi::DenseTensor& tensor,
                       const phi::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is,
                           phi::DenseTensor* tensor,
                           const phi::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is,
                           phi::DenseTensor* tensor,
                           const phi::DeviceContext& dev_ctx,
                           const size_t& seek,
                           const std::vector<int64_t>& shape);

TEST_API LoD ConvertToOffsetBasedLoD(const LoD& length_lod);

void SerializeToStream(std::ostream& os, const phi::DenseTensor& tensor);

void DeserializeFromStream(std::istream& os, phi::DenseTensor* tensor);

}  // namespace framework
}  // namespace paddle
