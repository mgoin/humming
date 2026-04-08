import functools
from typing import TYPE_CHECKING

import torch

from humming.config import GemmType
from humming.tune.base import DeviceHeuristics
from humming.tune.sm8x import (
    Sm80Heuristics,
    Sm86Heuristics,
    Sm87Heuristics,
    Sm89Heuristics,
)
from humming.tune.sm75 import Sm75Heuristics
from humming.tune.sm90 import Sm90Heuristics

if TYPE_CHECKING:
    from humming.layer import HummingLayerMeta

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
}


def get_heuristics_class(
    sm_version: int | tuple[int, int] | None = None,
    device: int | torch.device | None = None,
) -> type[DeviceHeuristics]:
    if sm_version is None:
        sm_version = torch.cuda.get_device_capability(device)
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int)

    return heuristics_map[sm_version]


@functools.lru_cache(maxsize=1024)
def get_heuristics_config(
    meta: "HummingLayerMeta | dict",
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    gemm_type: str | GemmType = "dense",
):
    from humming.layer import HummingLayerMeta

    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    if isinstance(meta, dict):
        meta = HummingLayerMeta(**meta)
    heuristics_cls = get_heuristics_class()
    if isinstance(shape_m, int):
        return heuristics_cls.get_config(
            meta=meta,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
    else:
        return heuristics_cls.get_configs(
            meta=meta,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
    )


def get_default_moe_block_size_configs(
    meta: "HummingLayerMeta | dict",
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
):
    assert meta.num_experts is not None and meta.num_experts > 0
    kernel_configs = get_heuristics_config(
        meta=meta,
        use_f16_accum=use_f16_accum,
        use_batch_invariant=use_batch_invariant,
    )
    block_size_configs = []
    for min_shape_m, max_shape_m, config in kernel_configs:
        block_size_configs += [min_shape_m, max_shape_m, config["block_shape"][0]]
    return block_size_configs
