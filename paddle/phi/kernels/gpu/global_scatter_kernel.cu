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

#include "paddle/phi/core/distributed/collective/process_group.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/collective/process_group_nccl.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename Context, typename T>
struct GlobalScatterFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x_in,
                  const DenseTensor& local_count_in,
                  const DenseTensor& global_count_in,
                  int ring_id,
                  bool use_calc_stream,
                  DenseTensor* out);
};

template <typename Context, typename T>
struct GlobalScatterProcessGroupFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& x_in,
                  const DenseTensor& local_count_in,
                  const DenseTensor& global_count_in,
                  int ring_id,
                  bool use_calc_stream,
                  DenseTensor* out);
};

template <typename T>
struct GlobalScatterFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& x_in,
                  const DenseTensor& local_count_in,
                  const DenseTensor& global_count_in,
                  int ring_id,
                  bool use_calc_stream UNUSED,
                  DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = &x_in;
    auto local_count = &local_count_in;
    auto global_count = &global_count_in;

    auto local_count_type = local_count->dtype();
    auto global_count_type = global_count->dtype();
    if (local_count_type != phi::DataType::INT64) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != phi::DataType::INT64) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }

    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    phi::DenseTensor cpu_local_count;
    if (local_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_local_count_data = local_count->data<int64_t>();
    } else {
      phi::Copy(dev_ctx, *local_count, phi::CPUPlace(), true, &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
    }
    auto global_count_len = 0;
    phi::DenseTensor cpu_global_count;
    if (global_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_global_count_data = global_count->data<int64_t>();
      global_count_len = global_count->numel();
    } else {
      phi::Copy(
          dev_ctx, *global_count, phi::CPUPlace(), true, &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
      global_count_len = cpu_global_count.numel();
    }

    ncclDataType_t dtype = phi::ToNCCLDataType(x->dtype());

    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for global scatter op must be non-negative.",
            ring_id));

    gpuStream_t stream = nullptr;
    stream = dev_ctx.stream();
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;
    int nranks = 0;

    comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
        dev_ctx.GetCommContext());
    PADDLE_ENFORCE_NE(comm_ctx,
                      nullptr,
                      common::errors::Unavailable(
                          "NCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

    nranks = comm_ctx->GetSize();

    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;
    int64_t fwd_count = 0;

    for (auto i = 0; i < global_count_len; ++i) {
      fwd_count += cpu_global_count_data[i];
    }
    phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }

    auto recv_ptr = 0;
    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out);

    for (auto i = 0; i < n_expert; ++i) {
      comm_ctx->GroupStart();
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_local_count_data[idx]) {
          auto send_buf = distributed::GetPartialTensor(
              *x,
              expert_ptr[idx] * in_feat,
              cpu_local_count_data[idx] * in_feat);

          comm_ctx->Send(
              send_buf, cpu_local_count_data[idx] * in_feat, j, stream);
        }
        if (cpu_global_count_data[idx]) {
          auto recv_buf = distributed::GetPartialTensor(
              *out, recv_ptr * in_feat, cpu_global_count_data[idx] * in_feat);
          comm_ctx->Recv(
              &recv_buf, cpu_global_count_data[idx] * in_feat, j, stream);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
      comm_ctx->GroupEnd();
    }
#else
    PADDLE_THROW(
        common::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        common::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

template <typename T>
struct GlobalScatterProcessGroupFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& dev_ctx,
                  const DenseTensor& x_in,
                  const DenseTensor& local_count_in,
                  const DenseTensor& global_count_in,
                  int ring_id,
                  bool use_calc_stream UNUSED,
                  DenseTensor* out) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = &x_in;
    auto local_count = &local_count_in;
    auto global_count = &global_count_in;

    auto local_count_type = local_count->dtype();
    auto global_count_type = global_count->dtype();
    if (local_count_type != phi::DataType::INT64) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != phi::DataType::INT64) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }

    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    phi::DenseTensor cpu_local_count;
    if (local_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_local_count_data = local_count->data<int64_t>();
    } else {
      phi::Copy(dev_ctx, *local_count, phi::CPUPlace(), true, &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
    }
    auto global_count_len = 0;
    phi::DenseTensor cpu_global_count;
    if (global_count->place().GetType() == phi::AllocationType::CPU) {
      cpu_global_count_data = global_count->data<int64_t>();
      global_count_len = global_count->numel();
    } else {
      phi::Copy(
          dev_ctx, *global_count, phi::CPUPlace(), true, &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
      global_count_len = cpu_global_count.numel();
    }

    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for global scatter op must be non-negative.",
            ring_id));

    auto map = phi::distributed::ProcessGroupMapFromGid::getInstance();
    phi::distributed::ProcessGroup* pg = map->get(ring_id);
    int nranks = pg->GetSize();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;
    int64_t fwd_count = 0;

    for (auto i = 0; i < global_count_len; ++i) {
      fwd_count += cpu_global_count_data[i];
    }
    phi::DDim out_dims = common::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }

    auto recv_ptr = 0;
    out->Resize(out_dims);
    dev_ctx.template Alloc<T>(out);

    for (auto i = 0; i < n_expert; ++i) {
      paddle::distributed::ProcessGroupNCCL::GroupStart();
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_local_count_data[idx]) {
          phi::DenseTensor tmp = *x;
          pg->Send(tmp,
                   j,
                   expert_ptr[idx] * in_feat,
                   cpu_local_count_data[idx] * in_feat,
                   /*sync_op*/ true);
        }
        if (cpu_global_count_data[idx]) {
          pg->Recv(out,
                   j,
                   recv_ptr * in_feat,
                   cpu_global_count_data[idx] * in_feat,
                   /*sync_op*/ true);
          recv_ptr += cpu_global_count_data[idx];
        }
      }
      paddle::distributed::ProcessGroupNCCL::GroupEnd();
    }

#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

#else
    PADDLE_THROW(
        common::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        common::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

template <typename T, typename Context>
void GlobalScatterKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& local_count,
                         const DenseTensor& global_count,
                         int ring_id,
                         bool use_calc_stream,
                         DenseTensor* out) {
  const int rid = ring_id;
  auto map = phi::distributed::ProcessGroupMapFromGid::getInstance();
  if (map->has(rid)) {
    GlobalScatterProcessGroupFunctor<phi::GPUContext, T> functor_;
    functor_(
        dev_ctx, x, local_count, global_count, ring_id, use_calc_stream, out);
  } else {
    GlobalScatterFunctor<phi::GPUContext, T> functor_;
    functor_(
        dev_ctx, x, local_count, global_count, ring_id, use_calc_stream, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(global_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::GlobalScatterKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
}
