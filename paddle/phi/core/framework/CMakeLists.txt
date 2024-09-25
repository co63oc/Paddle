set(FRAMEWORK_SRCS "")

proto_library(framework_proto SRCS framework.proto)

list(
  APPEND
  FRAMEWORK_SRCS
  data_type.cc
  dlpack_tensor.cc
  lod_tensor.cc
  selected_rows_utils.cc
  string_array.cc
  tensor_util.cc
  version.cc)

collect_srcs(core_srcs SRCS ${FRAMEWORK_SRCS})
