import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming import dtypes
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


@dataclasses.dataclass(kw_only=True)
class QuantWeightKernel(KernelRuntime):
    name: ClassVar[str] = "quant_weight"
    source_dtype: dtypes.DataType
    target_dtype: dtypes.DataType
    group_size: int
    has_scale: bool
    use_e8m0_scale: bool
    has_zero_point: bool = False
    is_fp_zero_point: bool = False

    def __post_init__(self):
        self.code = CODE_TEMPLATE.render(
            source_dtype=self.source_dtype.to_cpp_str(),
            target_dtype=self.target_dtype.to_cpp_str(),
            group_size=self.group_size,
            has_scale=int(self.has_scale),
            use_e8m0_scale=int(self.use_e8m0_scale),
            has_zero_point=int(self.has_zero_point),
            is_fp_zero_point=int(self.is_fp_zero_point),
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
        self.check_context()
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
