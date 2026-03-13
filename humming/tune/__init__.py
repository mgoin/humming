import torch

from humming.tune.base import DeviceHeuristics
from humming.tune.sm89 import Sm89Heuristics


def get_heuristics_class(
    sm_version: int | tuple[int, int] | None = None,
    device: int | torch.device | None = None,
) -> type[DeviceHeuristics]:
    if sm_version is None:
        sm_version = torch.cuda.get_device_capability(device)
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int)

    if sm_version == 89:
        return Sm89Heuristics

    return Sm89Heuristics
