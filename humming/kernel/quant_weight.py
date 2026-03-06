import ctypes

import cuda.bindings.driver as cbd
import jinja2
from humming import dtypes
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/quant_weight.cuh>

auto ptr = reinterpret_cast<void*>(&quant_weight<
    {{source_dtype}},
    {{target_dtype}},
    {{group_size}},
    {{has_scale}},
    {{use_e8m0_scale}},
    {{has_zero_point}}
  >);
""")


class QuantWeightKernel(KernelRuntime):
    name = "quant_weight"

    def __init__(
        self,
        source_dtype,
        target_dtype,
        group_size,
        has_scale,
        use_e8m0_scale,
        has_zero_point=False,
    ):
        if self.inited:
            return
        self.group_size = group_size
        self.has_scale = has_scale
        self.use_e8m0_scale = use_e8m0_scale
        self.has_zero_point = has_zero_point
        self.code = CODE_TEMPLATE.render(
            source_dtype=source_dtype.to_cpp_str(),
            target_dtype=target_dtype.to_cpp_str(),
            group_size=group_size,
            has_scale=int(has_scale),
            use_e8m0_scale=int(use_e8m0_scale),
            has_zero_point=int(has_zero_point),
        )
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        scales: torch.Tensor | None,
        zero_point: torch.Tensor | None,
    ):
        group_size = self.group_size
        group_size = inputs.size(-1) if group_size <= 0 else group_size

        device = inputs.device
        config = cbd.CUlaunchConfig()
        config.gridDimX = inputs.nelement() // group_size
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            0 if scales is None else scales.data_ptr(),
            0 if zero_point is None else zero_point.data_ptr(),
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)


def humming_quant_weight(
    inputs: torch.Tensor,
    source_dtype_str: str,
    target_dtype_str: str,
    group_size: int,
    has_scale: bool,
    use_e8m0_scale: bool,
    has_zero_point: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    group_size = inputs.size(-1) if group_size <= 0 else group_size
    source_dtype = dtypes.DataType.from_str(source_dtype_str)
    target_dtype = dtypes.DataType.from_str(target_dtype_str)

    assert inputs.is_cuda
    assert inputs.is_contiguous()
    outputs = torch.empty_like(inputs, dtype=torch.int32)

    if has_scale:
        scale_shape = inputs.shape[:-1] + (inputs.size(-1) // group_size,)
        scale_dtype = torch.float8_e8m0fnu if use_e8m0_scale else torch.float32
        scales = torch.empty(scale_shape, device=inputs.device, dtype=scale_dtype)
        zero_point = torch.empty(scale_shape, device=inputs.device, dtype=torch.int32)
    else:
        scales = torch.empty(0)

    if has_scale and has_zero_point:
        zero_point = torch.empty(scale_shape, device=inputs.device, dtype=torch.int32)
    else:
        zero_point = torch.empty(0)

    kernel = QuantWeightKernel(
        source_dtype=source_dtype,
        target_dtype=target_dtype,
        group_size=group_size,
        has_scale=has_scale,
        has_zero_point=has_zero_point,
        use_e8m0_scale=use_e8m0_scale,
    )
    kernel(inputs=inputs, outputs=outputs, scales=scales, zero_point=zero_point)

    return outputs, scales, zero_point
