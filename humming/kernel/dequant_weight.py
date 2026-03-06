import ctypes
import math

import cuda.bindings.driver as cbd
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = """
#include <humming/kernel/dequant_weight.cuh>

auto ptr = reinterpret_cast<void*>(&dequant_unpacked_fp_type);
"""


class DequantKernel(KernelRuntime):
    name = "dequant_unpacked_fp_type"

    def __init__(self):
        if self.inited:
            return
        self.code = CODE_TEMPLATE
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
        )
        self.prepare()
        self.inited = True

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        exponent_bits: int,
        mantissa_bits: int,
        is_signed: bool,
    ):
        device = inputs.device
        total_size = inputs.nelement()
        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(total_size / (32 * 32))
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            total_size,
            exponent_bits,
            mantissa_bits,
            is_signed,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)
        return outputs


def humming_dequant_weight(
    inputs: torch.Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    is_signed: bool,
) -> torch.Tensor:
    assert inputs.dtype == torch.int32
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    outputs = torch.empty_like(inputs, dtype=torch.float32)

    kernel = DequantKernel()
    kernel(
        inputs=inputs,
        outputs=outputs,
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        is_signed=is_signed,
    )

    return outputs
