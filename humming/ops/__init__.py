from typing import TYPE_CHECKING

import torch

from humming.ops.bench import tops_bench  # noqa
from humming.ops.input import quant_input
from humming.ops.utils import init_humming_launcher, register_op
from humming.ops.weight import (
    dequant_weight,
    pack_weight,
    quant_weight,
    repack_weight,
    unpack_weight,
)


def register_kernel(cubin_path: str, func_name: str) -> int:
    init_humming_launcher()
    return torch.ops.humming.register_kernel(cubin_path, func_name)


def launch_kernel(
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
    init_humming_launcher()
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


register_op("humming::quant_input", quant_input, quant_input)
register_op("humming::quant_weight", quant_weight, quant_weight)
register_op("humming::dequant_weight", dequant_weight, dequant_weight)
register_op("humming::repack_weight", repack_weight, repack_weight)
register_op("humming::pack_weight", pack_weight, pack_weight)
register_op("humming::unpack_weight", unpack_weight, unpack_weight)


if not TYPE_CHECKING:
    quant_input = torch.ops.humming.quant_input
    quant_weight = torch.ops.humming.quant_weight
    dequant_weight = torch.ops.humming.dequant_weight
    repack_weight = torch.ops.humming.repack_weight
    pack_weight = torch.ops.humming.pack_weight
    unpack_weight = torch.ops.humming.unpack_weight
