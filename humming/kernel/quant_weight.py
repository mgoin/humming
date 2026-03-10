import ctypes

import cuda.bindings.driver as cbd
import jinja2
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
    {{has_zero_point}},
    {{is_fp_zero_point}}
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
        is_fp_zero_point=False,
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
            is_fp_zero_point=int(is_fp_zero_point),
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
