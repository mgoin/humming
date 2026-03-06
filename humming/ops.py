import os
from typing import TYPE_CHECKING, Callable

import torch
import torch.utils.cpp_extension

from humming.kernel.dequant_weight import humming_dequant_weight
from humming.kernel.pack_weight import humming_pack_weight
from humming.kernel.quant_input import humming_quant_input
from humming.kernel.quant_weight import humming_quant_weight
from humming.kernel.repack_weight import humming_repack_weight
from humming.kernel.unpack_weight import humming_unpack_weight

_libs = {}
_launcher_inited = False


def _register_op(
    name: str,
    impl_func: Callable,
    fake_impl_func: Callable | None = None,
    mutates_args: list[str] | None = None,
):
    mutates_args = [] if mutates_args is None else mutates_args
    schema_str = torch.library.infer_schema(impl_func, mutates_args=mutates_args)
    lib_name, op_name = name.split("::")

    if lib_name not in _libs:
        _lib = torch.library.Library(lib_name, "FRAGMENT")
        _libs[lib_name] = _lib

    _lib = _libs[lib_name]
    _lib.define(op_name + schema_str)
    _lib.impl(op_name, impl_func, dispatch_key="CUDA")
    if fake_impl_func is not None:
        _lib._register_fake(op_name, fake_impl_func)


def _init_humming_launcher():
    global _launcher_inited
    if _launcher_inited:
        return

    import humming

    dirname = os.path.dirname(humming.__file__)
    filename = os.path.join(dirname, "csrc/launcher/launcher.cpp")
    filename = os.path.abspath(filename)

    torch.utils.cpp_extension.load(
        name="humming_extension",
        sources=[filename],
        extra_include_paths=["/usr/local/cuda/include"],
        extra_ldflags=[
            "-lcuda",
            "-L/usr/local/cuda/lib64",
            "-lc10_cuda",
            "-ltorch_cuda",
        ],
        extra_cflags=["-O3"],
    )

    _launcher_inited = True


def humming_register_kernel(cubin_path: str, func_name: str) -> int:
    _init_humming_launcher()
    return torch.ops.humming.register_kernel(cubin_path, func_name)


def humming_launch_kernel(
    configs: list[int],
    inputs: torch.Tensor,
    weight: torch.Tensor,
    outputs: torch.Tensor | None = None,
    input_scale: torch.Tensor | None = None,
    weight_scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    sorted_token_ids: torch.Tensor | None = None,
    expert_ids: torch.Tensor | None = None,
    num_tokens_padded: torch.Tensor | None = None,
    locks: torch.Tensor | None = None,
) -> torch.Tensor:
    _init_humming_launcher()
    return torch.ops.humming.launch_kernel(
        configs,
        inputs,
        weight,
        outputs,
        input_scale,
        weight_scale,
        zero_point,
        bias,
        global_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_padded,
        locks,
    )


_register_op("humming::quant_input", humming_quant_input, humming_quant_input)
_register_op("humming::quant_weight", humming_quant_weight, humming_quant_weight)
_register_op("humming::dequant_weight", humming_dequant_weight, humming_dequant_weight)
_register_op("humming::repack_weight", humming_repack_weight, humming_repack_weight)
_register_op("humming::pack_weight", humming_pack_weight, humming_pack_weight)
_register_op("humming::unpack_weight", humming_unpack_weight, humming_unpack_weight)


if not TYPE_CHECKING:
    humming_quant_input = torch.ops.humming.quant_input
    humming_quant_weight = torch.ops.humming.quant_weight
    humming_dequant_weight = torch.ops.humming.dequant_weight
    humming_repack_weight = torch.ops.humming.repack_weight
    humming_pack_weight = torch.ops.humming.pack_weight
    humming_unpack_weight = torch.ops.humming.unpack_weight
