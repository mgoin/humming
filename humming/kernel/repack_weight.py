import ctypes
import math

import cuda.bindings.driver as cbd
import jinja2
import torch

from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/process.cuh>

auto ptr = reinterpret_cast<void*>(&weight_repack_nk<
    {{weight_bits}},
    {{activation_bits}},
    {{is_weight_pakced}},
    {{should_preprocess_for_int2fp}},
    {{should_preprocess_with_zp}},
    {{group_size_zp}}
  >);
""")


class RepackWeightKernel(KernelRuntime):
    name = "weight_repack_nk"

    def __init__(
        self,
        weight_bits: int,
        activation_bits: int,
        is_weight_pakced: bool,
        should_preprocess_for_int2fp: bool = False,
        should_preprocess_with_zp: bool = False,
        group_size_zp: int = 0,
    ):
        if self.inited:
            return
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.is_weight_pakced = is_weight_pakced
        self.should_preprocess_with_zp = should_preprocess_with_zp
        self.group_size_zp = group_size_zp

        if self.should_preprocess_with_zp:
            assert should_preprocess_for_int2fp

        self.code = CODE_TEMPLATE.render(
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            is_weight_pakced=int(is_weight_pakced),
            should_preprocess_for_int2fp=int(should_preprocess_for_int2fp),
            should_preprocess_with_zp=int(should_preprocess_with_zp),
            group_size_zp=group_size_zp,
        )
        self.arg_types = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        self.prepare()

    def __call__(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        zero_point: torch.Tensor | None,
        padded_shape_n: int | None = None,
        padded_shape_k: int | None = None,
    ):
        num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
        shape_n = inputs.size(-2)
        shape_k = inputs.size(-1)
        device = inputs.device

        config = cbd.CUlaunchConfig()
        config.gridDimX = math.ceil(shape_n / 64)
        config.gridDimY = math.ceil(shape_k / 64)
        config.gridDimZ = num_experts
        config.blockDimX = 32
        config.blockDimY = 1
        config.blockDimZ = 1
        config.hStream = torch.cuda.current_stream(device).cuda_stream

        arg_values = (
            inputs.data_ptr(),
            outputs.data_ptr(),
            0 if zero_point is None else zero_point.data_ptr(),
            shape_n,
            shape_k,
            padded_shape_n or shape_n,
            padded_shape_k or shape_k,
        )

        cbd.cuLaunchKernelEx(config, self.kernel, (arg_values, self.arg_types), 0)


def humming_repack_weight(
    inputs: torch.Tensor,
    weight_bits: int,
    activation_bits: int,
    is_weight_pakced: bool,
    should_preprocess_for_int2fp: bool = False,
    should_preprocess_with_zp: bool = False,
    group_size_zp: int = 0,
    padded_shape_n: int | None = None,
    padded_shape_k: int | None = None,
    zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    assert inputs.ndim in [2, 3]
    assert inputs.is_cuda
    assert inputs.is_contiguous()
    assert inputs.dtype == torch.int32
    device = inputs.device
    num_experts = 1 if inputs.ndim == 2 else inputs.size(0)
    shape_n = inputs.size(-2)
    shape_k = inputs.size(-1)
    if is_weight_pakced:
        assert shape_k * 32 % weight_bits == 0
        shape_k = shape_k * 32 // weight_bits

    if should_preprocess_with_zp:
        assert zero_point is not None
        group_size_zp = shape_k if group_size_zp == 0 else group_size_zp
        zero_point_shape = inputs.shape[:-1] + (math.ceil(shape_k / group_size_zp),)

        if is_weight_pakced:
            assert shape_n * weight_bits % 32 == 0
            packed_shape_n = shape_n * weight_bits // 32
            zero_point_shape = zero_point_shape[:-2] + (packed_shape_n,) + zero_point_shape[-1:]

        assert zero_point.shape == zero_point_shape

    pack_size_k = 256 // activation_bits
    output_shape = (
        shape_k // pack_size_k,
        shape_n * pack_size_k * weight_bits // 32,
    )
    if inputs.ndim == 3:
        output_shape = (num_experts,) + output_shape

    outputs = torch.empty(output_shape, dtype=torch.int32, device=device)

    kernel = RepackWeightKernel(
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        is_weight_pakced=is_weight_pakced,
        should_preprocess_for_int2fp=should_preprocess_for_int2fp,
        should_preprocess_with_zp=should_preprocess_with_zp,
        group_size_zp=group_size_zp,
    )

    kernel(
        inputs=inputs,
        outputs=outputs,
        zero_point=zero_point,
        padded_shape_n=padded_shape_n,
        padded_shape_k=padded_shape_k,
    )

    return outputs
