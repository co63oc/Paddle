set(FRAMEWORK_SRCS "")

proto_library(framework_proto SRCS framework.proto)

list(
  APPEND
  FRAMEWORK_SRCS
  attribute.cc
  data_type.cc
  dlpack_tensor.cc
  reader.cc
  lod_rank_table.cc
  lod_tensor.cc
  scope.cc
  selected_rows_utils.cc
  string_array.cc
  tensor_util.cc
  type_defs.cc
  var_type_traits.cc
  variable_helper.cc
  version.cc)

collect_srcs(core_srcs SRCS ${FRAMEWORK_SRCS})
