/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/common/layout.h"
#include "paddle/phi/core/framework/data_type.h"
#include "paddle/phi/core/utils/data_type.h"

// TODO(chenweihang): this file may need to be removed

namespace paddle {
namespace framework {

using DataType = phi::DataType;
using DataLayout = phi::DataLayout;

using phi::DataTypeToString;
using phi::SizeOf;
using phi::TransToPhiDataType;

inline proto::VarType::Type TransToProtoVarType(const DataType& dtype) {
  return static_cast<proto::VarType::Type>(phi::TransToProtoVarType(dtype));
}

}  // namespace framework
}  // namespace paddle
