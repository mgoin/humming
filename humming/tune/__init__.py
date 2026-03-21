import functools

import torch

from humming.layer import HummingLayerMeta
from humming.tune.base import DeviceHeuristics
from humming.tune.sm8x import Sm80Heuristics, Sm86Heuristics, Sm87Heuristics, Sm89Heuristics
from humming.tune.sm75 import Sm75Heuristics
from humming.tune.sm90 import Sm90Heuristics

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
def get_heuristics_config(meta: HummingLayerMeta, use_stream_k: bool, use_f16_accum: bool):
    heuristics_cls = get_heuristics_class()
    return heuristics_cls.get_configs(meta, use_stream_k, use_f16_accum)
