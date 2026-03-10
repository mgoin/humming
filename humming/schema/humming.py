import dataclasses
from typing import Any, Literal

import torch

from humming import dtypes
from humming.schema.base import BaseInputSchema, BaseWeightSchema


@dataclasses.dataclass(kw_only=True)
class HummingWeightSchema(BaseWeightSchema):
    quant_method: str = "humming"
    b_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType | None = None
    weight_scale_group_size: int
    has_global_scale: bool = False
    has_zero_point: bool = False
    is_fp_zero_point: bool = False

    TENSOR_NAMES = Literal["weight", "weight_scale", "zero_point", "global_scale", "bias"]

    def __post_init__(self):
        assert self.quant_method == "humming"
        if isinstance(self.b_dtype, str):
            self.b_dtype = dtypes.DataType.from_str(str(self.b_dtype))
        if isinstance(self.bs_dtype, str):
            self.bs_dtype = dtypes.DataType.from_str(str(self.bs_dtype))

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.b_dtype.num_bits
        group_size = self.weight_scale_group_size or shape_k
        scale_type = "group" if self.weight_scale_group_size > 0 else "channel"

        scale_torch_dtype = param_dtype
        if self.bs_dtype == dtypes.float8e8m0:
            scale_torch_dtype = torch.float8_e8m0fnu
        elif self.bs_dtype == dtypes.float8e4m3:
            scale_torch_dtype = torch.float8_e4m3fn
        elif self.bs_dtype == dtypes.float8e5m2:
            scale_torch_dtype = torch.float8_e5m2

        tensor_meta = {
            "weight": {
                "shape": (shape_n, shape_k * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            },
            "weight_scale": {
                "shape": (shape_n, shape_k // group_size),
                "dtype": scale_torch_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0, "scale_type": scale_type},
            },
        }

        if self.has_zero_point and not self.is_fp_zero_point:
            tensor_meta["zero_point"] = {
                "shape": (shape_n * num_bits // 32, shape_k // group_size),
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            }

        if self.has_zero_point and self.is_fp_zero_point:
            tensor_meta["zero_point"] = {
                "shape": (shape_n, shape_k // group_size),
                "dtype": param_dtype,
                "extra_attrs": {"input_dim": 1, "output_dim": 0},
            }

        if self.has_global_scale:
            tensor_meta["global_scale"] = {
                "shape": (1,),
                "dtype": torch.float32,
                "extra_attrs": {"scale_type": "tensor"},
            }

        if has_bias:
            tensor_meta["bias"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0},
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        num_bits = self.b_dtype.num_bits
        shape_n = tensors["weight"].size(-2)
        shape_k = tensors["weight"].size(-1) * 32 // num_bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple["HummingWeightSchema", dict[str, torch.Tensor]]:
        return self, tensors


@dataclasses.dataclass(kw_only=True)
class HummingInputSchema(BaseInputSchema):
    quant_method: str = "humming"
    a_dtype: dtypes.DataType | None = None
    input_scale_group_size: int = 0

    def __post_init__(self):
        if isinstance(self.a_dtype, str):
            self.a_dtype = dtypes.DataType.from_str(str(self.a_dtype))

    def get_activation_bits(self):
        if self.a_dtype is None:
            return 16
        return self.a_dtype.num_bits

    def get_tensors_attrs(
        self,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        return {}

    def convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple["HummingInputSchema", dict[str, torch.Tensor]]:
        return self, {}
