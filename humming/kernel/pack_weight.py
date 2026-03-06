import ctypes

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/pack_weight.cuh>

auto ptr = reinterpret_cast<void*>(&pack_weight_kernel<{{num_bits}}>);
""")


class PackWeightKernel(KernelRuntime):
    name = "pack_weight"

    def __init__(self, num_bits):
        if self.inited:
            return
        self.num_bits = num_bits
        self.code = CODE_TEMPLATE.render(num_bits=num_bits)
        self.arg_types = (ctypes.c_void_p, ctypes.c_void_p)
        self.prepare()

    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor):
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


def humming_pack_weight(inputs: torch.Tensor, num_bits: int) -> torch.Tensor:
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.nelement() % (32 * 32) == 0
    assert inputs.size(-1) * num_bits % 32 == 0
    assert inputs.dtype == torch.int32

    output_shape = inputs.shape[:-1] + (inputs.size(-1) * num_bits // 32,)
    outputs = torch.empty(output_shape, dtype=torch.int32, device=inputs.device)

    kernel = PackWeightKernel(num_bits)
    return kernel(inputs=inputs, outputs=outputs)
