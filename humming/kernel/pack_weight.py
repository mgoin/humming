import ctypes
import dataclasses
from typing import ClassVar

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/pack_weight.cuh>

auto ptr = reinterpret_cast<void*>(&pack_weight_kernel<{{num_bits}}>);
""")


@dataclasses.dataclass(kw_only=True)
class PackWeightKernel(KernelRuntime):
    name: ClassVar[str] = "pack_weight"
    num_bits: int

    def __post_init__(self):
        self.code = CODE_TEMPLATE.render(num_bits=self.num_bits)
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        self.check_context()
        device = inputs.device
        config = cbd.CUlaunchConfig()
        config.gridDimX = inputs.nelement() // (32 * 32)
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (inputs.data_ptr(), outputs.data_ptr())

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
