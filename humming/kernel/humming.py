import dataclasses
import os
import zlib
from typing import Any, ClassVar

import jinja2

import humming.jit.utils as jit_utils
from humming import dtypes
from humming.config import (
    EpilogueConfig,
    MmaConfig,
    MmaOpClass,
    MmaType,
    MoEConfig,
    PipelineConfig,
    QuantParamConfig,
    SchedulerConfig,
)
from humming.jit.runtime import KernelRuntime

CODE_TEMPLATE = jinja2.Template("""
#if {{use_warp_spec}}
#include <humming/kernel/humming_ws.cuh>
#else
#include <humming/kernel/humming.cuh>
#endif

{{custom_activation_func}}

class MmaOpClass {
public:
{{mma_op_class}}
};

class SchedulerConfig {
public:
{{scheduler_config}}
};

class PipelineConfig {
public:
{{pipeline_config}}
};

class EpilogueConfig {
public:
{{epilogue_config}}
};

class QuantParamConfig {
public:
{{quant_param_config}}
};

class MoEConfig {
public:
{{moe_config}}
};

using SharedStorageType = SharedStorage<
    MmaOpClass,
    Shape<{{block_shape[0]}}, {{block_shape[1]}}, {{block_shape[2]}}>,
    Shape<{{warp_shape[0]}}, {{warp_shape[1]}}, {{warp_shape[2]}}>,
    {{a_dtype}},
    {{b_dtype}},
    {{bs_dtype}},
    PipelineConfig,
    EpilogueConfig,
    QuantParamConfig,
    MoEConfig>;

auto ptr = reinterpret_cast<void*>(&humming<
    MmaOpClass,
    Shape<0, {{problem_shape[1]}}, {{problem_shape[2]}}>,
    Shape<{{block_shape[0]}}, {{block_shape[1]}}, {{block_shape[2]}}>,
    Shape<{{warp_shape[0]}}, {{warp_shape[1]}}, {{warp_shape[2]}}>,
    Shape<0, {{pad_shape[1]}}, {{pad_shape[2]}}>,
    {{a_dtype}},
    {{b_dtype}},
    {{c_dtype}},
    {{bs_dtype}},
    SchedulerConfig,
    PipelineConfig,
    EpilogueConfig,
    QuantParamConfig,
    MoEConfig>);


extern "C" __constant__ uint32_t SMEM_SIZE = sizeof(SharedStorageType);
extern "C" __constant__ uint32_t SMEM_SIZE_A = sizeof(SharedStorageType::a);
extern "C" __constant__ uint32_t SMEM_SIZE_B = sizeof(SharedStorageType::b);
extern "C" __constant__ uint32_t SMEM_SIZE_REDUCE = sizeof(SharedStorageType::reduce);

extern "C" __constant__ uint32_t PROBLEM_SHAPE_N = {{problem_shape[1]}};
extern "C" __constant__ uint32_t PROBLEM_SHAPE_K = {{problem_shape[2]}};

extern "C" __constant__ uint32_t BLOCK_SHAPE_M = {{block_shape[0]}};
extern "C" __constant__ uint32_t BLOCK_SHAPE_N = {{block_shape[1]}};
extern "C" __constant__ uint32_t BLOCK_SHAPE_K = {{block_shape[2]}};

extern "C" __constant__ uint32_t WARP_SHAPE_M = {{warp_shape[0]}};
extern "C" __constant__ uint32_t WARP_SHAPE_N = {{warp_shape[1]}};
extern "C" __constant__ uint32_t WARP_SHAPE_K = {{warp_shape[2]}};

extern "C" __constant__ uint32_t PAD_SHAPE_N = {{pad_shape[1]}};
extern "C" __constant__ uint32_t PAD_SHAPE_K = {{pad_shape[2]}};

extern "C" __constant__ uint32_t A_DTYPE_ID = {{a_dtype}}::kId;
extern "C" __constant__ uint32_t B_DTYPE_ID = {{b_dtype}}::kId;
extern "C" __constant__ uint32_t C_DTYPE_ID = {{c_dtype}}::kId;
extern "C" __constant__ uint32_t BS_DTYPE_ID = {{bs_dtype}}::kId;

extern "C" __constant__ uint32_t IS_GLU_ACTIVATION = {{is_glu_activation}};

{{scheduler_config_extern}}

{{pipeline_config_extern}}

{{epilogue_config_extern}}

{{quant_param_config_extern}}

{{moe_config_extern}}

""")


@dataclasses.dataclass(kw_only=True)
class HummingKernel(
    KernelRuntime,
    SchedulerConfig,
    PipelineConfig,
    EpilogueConfig,
    QuantParamConfig,
    MoEConfig,
    MmaConfig,
):
    name: ClassVar[str] = "humming"
    problem_shape: tuple[int, int, int]
    block_shape: tuple[int, int, int]
    warp_shape: tuple[int, int, int]
    pad_shape: tuple[int, int, int] = (0, 0, 0)
    a_dtype: dtypes.DataType
    b_dtype: dtypes.DataType
    c_dtype: dtypes.DataType
    bs_dtype: dtypes.DataType

    def init_kernel(self) -> None:
        for key in ["a_dtype", "b_dtype", "c_dtype", "bs_dtype"]:
            dtype = getattr(self, key)
            if isinstance(dtype, str):
                setattr(self, key, dtypes.DataType.from_str(dtype))

        self.scheduler_config = SchedulerConfig.from_dict(vars(self))
        self.pipeline_config = PipelineConfig.from_dict(vars(self))
        self.epilogue_config = EpilogueConfig.from_dict(vars(self))
        self.quant_param_config = QuantParamConfig.from_dict(vars(self))
        self.moe_config = MoEConfig.from_dict(vars(self))
        self.mma_config = MmaConfig.from_dict(vars(self))
        name_list = ["scheduler", "pipeline", "epilogue", "quant_param", "moe", "mma"]
        for name in name_list:
            config = getattr(self, name + "_config")
            for key, value in vars(config).items():
                setattr(self, key, value)

        self.check_shape()
        self.check_dtype()
        self.check_config()
        self.mma_op_class = self.select_mma_op_class()

        custom_activation_func = self.epilogue_config.prepare_custom_activation_func()
        is_glu_activation = "_glu" in str(self.epilogue_config.activation_type).lower()
        assert isinstance(self.pipeline_config.use_warp_spec, bool)

        self.code = CODE_TEMPLATE.render(
            use_warp_spec=int(self.pipeline_config.use_warp_spec),
            mma_op_class=self.mma_op_class.to_cpp_str(),
            problem_shape=self.problem_shape,
            block_shape=self.block_shape,
            warp_shape=self.warp_shape,
            pad_shape=self.pad_shape,
            a_dtype=self.a_dtype.to_cpp_str(),
            b_dtype=self.b_dtype.to_cpp_str(),
            c_dtype=self.c_dtype.to_cpp_str(),
            bs_dtype=self.bs_dtype.to_cpp_str(),
            scheduler_config=self.scheduler_config.to_cpp_str(),
            pipeline_config=self.pipeline_config.to_cpp_str(),
            epilogue_config=self.epilogue_config.to_cpp_str(),
            quant_param_config=self.quant_param_config.to_cpp_str(),
            moe_config=self.moe_config.to_cpp_str(),
            scheduler_config_extern=self.scheduler_config.to_extern_cpp_str(),
            pipeline_config_extern=self.pipeline_config.to_extern_cpp_str(),
            epilogue_config_extern=self.epilogue_config.to_extern_cpp_str(),
            quant_param_config_extern=self.quant_param_config.to_extern_cpp_str(),
            moe_config_extern=self.moe_config.to_extern_cpp_str(),
            is_glu_activation=int(is_glu_activation),
            custom_activation_func=custom_activation_func,
        )

        self.prepare()

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        raise NotImplementedError

    def check_kwarg_keys(self, keys):
        def get_field_names(c):
            return [x.name for x in dataclasses.fields(c)]

        valid_keys = ["pad_shape"]
        valid_keys += get_field_names(SchedulerConfig)
        valid_keys += get_field_names(PipelineConfig)
        valid_keys += get_field_names(EpilogueConfig)
        valid_keys += get_field_names(QuantParamConfig)
        valid_keys += get_field_names(MoEConfig)
        valid_keys += get_field_names(MmaConfig)
        valid_keys = [x for x in valid_keys if not x.endswith("_threads")]
        invalid_keys = set(keys) - set(valid_keys)
        assert not invalid_keys, f"{invalid_keys}"

    def load_cubin(self):
        from humming import ops

        if self.cubin_loaded:
            return None
        kernel_filename = self.kernel_filename
        kernel_name = self.kernel_name
        self.kernel_id = ops.register_kernel(kernel_filename, kernel_name)
        self.kernel_dirname = os.path.dirname(kernel_filename)
        ref_kernel_id = zlib.crc32(kernel_filename.encode()) << 30
        ref_kernel_id += zlib.crc32(kernel_name.encode())
        assert ref_kernel_id == self.kernel_id
        module = jit_utils.make_humming_module("get_kernel_id", self.kernel_id)
        self.get_kernel_id = module.get_kernel_id

    def select_mma_op_class(self):
        if self.a_dtype in [dtypes.int4, dtypes.int8]:
            mma_cd_dtype = dtypes.int32
        elif self.mma_config.use_f16_accum:
            mma_cd_dtype = self.c_dtype
        else:
            mma_cd_dtype = dtypes.float32

        mma_shape_m = 64 if self.mma_config.mma_type == MmaType.WGMMA else 16
        mma_shape_n = self.warp_shape[0] if self.mma_config.mma_type == MmaType.WGMMA else 8
        mma_shape_k = 256 // self.a_dtype.num_bits
        if self.sm_version == 75 and self.a_dtype == dtypes.int8:
            mma_shape_m = 8

        if self.mma_config.mma_type == MmaType.MMA and self.warp_shape[0] % 16 == 8:
            mma_shape_m = 8

        if self.mma_config.mma_type == MmaType.MMA and mma_shape_m == 8:
            mma_shape_k = mma_shape_k // 2

        if self.mma_config.mma_type == MmaType.WGMMA:
            assert self.warp_shape[0] % mma_shape_n == 0
            assert self.warp_shape[1] % (mma_shape_m // 4) == 0
        else:
            assert self.warp_shape[0] % mma_shape_m == 0
            assert self.warp_shape[1] % mma_shape_n == 0
        assert self.warp_shape[2] % mma_shape_k == 0

        input_group_size = self.problem_shape[2]
        weight_group_size = self.problem_shape[2]
        scale_config = self.quant_param_config
        if scale_config.has_input_scale and scale_config.input_scale_group_size > 0:
            input_group_size = self.quant_param_config.input_scale_group_size
        if scale_config.has_weight_scale and scale_config.weight_scale_group_size > 0:
            weight_group_size = self.quant_param_config.weight_scale_group_size
        assert min(input_group_size, weight_group_size) >= mma_shape_k // 2
        if min(input_group_size, weight_group_size) == mma_shape_k // 2:
            mma_shape_k = mma_shape_k // 2

        return MmaOpClass.from_config(
            self.mma_config.mma_type,
            mma_shape_m,
            mma_shape_n,
            mma_shape_k,
            self.a_dtype,
            self.a_dtype,
            mma_cd_dtype,
        )

    def check_shape(self):
        assert self.problem_shape[1] % self.block_shape[1] == 0
        assert self.problem_shape[2] % self.block_shape[2] == 0
        assert self.block_shape[0] % self.warp_shape[0] == 0
        assert self.block_shape[1] % self.warp_shape[1] == 0
        assert self.block_shape[2] % self.warp_shape[2] == 0

        assert self.warp_shape[1] % 16 == 0
        assert jit_utils.is_power_of_two(self.block_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2])
        assert jit_utils.is_power_of_two(self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.warp_shape[2])
        assert jit_utils.is_power_of_two(self.block_shape[0] // self.warp_shape[0])
        assert jit_utils.is_power_of_two(self.block_shape[1] // self.warp_shape[1])
        assert jit_utils.is_power_of_two(self.block_shape[2] // self.warp_shape[2])
        assert self.problem_shape[1] > self.pad_shape[1]
        assert self.problem_shape[2] > self.pad_shape[2]
        assert self.pad_shape[1] % 8 == 0
        assert self.pad_shape[2] % (128 // self.a_dtype.num_bits) == 0

        assert self.warp_shape[1] <= 64
        if self.a_dtype.num_bits == 16:
            assert self.warp_shape[1] == 64
            assert self.warp_shape[2] >= 32
        elif self.a_dtype.num_bits == 8:
            assert self.warp_shape[1] >= 32
            assert self.warp_shape[2] >= 64
        elif self.a_dtype.num_bits == 4:
            assert self.warp_shape[1] >= 16
            assert self.warp_shape[2] >= 128

    def check_dtype(self):
        dtype_map = {
            dtypes.int4: 80,
            dtypes.int8: 75,
            dtypes.float4e2m1: 120,
            dtypes.float8e4m3: 89,
            dtypes.float8e5m2: 89,
            dtypes.bfloat16: 80,
            dtypes.float16: 75,
        }
        assert self.a_dtype in dtype_map
        assert self.sm_version >= dtype_map[self.a_dtype]
        assert self.b_dtype.num_bits <= 8
        assert self.b_dtype.num_bits <= self.a_dtype.num_bits
        if self.b_dtype.is_integer_type and self.a_dtype.is_integer_type:
            if self.a_dtype.num_bits == self.b_dtype.num_bits:
                assert self.a_dtype == self.b_dtype
            else:
                assert not self.b_dtype.is_signed
        elif self.b_dtype.is_integer_type and self.a_dtype.is_floating_point_type:
            assert not self.b_dtype.is_signed
            if self.quant_param_config.has_zero_point:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 1
            else:
                assert self.b_dtype.num_bits <= self.a_dtype.mantissa_bits + 2
        elif self.b_dtype.is_floating_point_type and self.a_dtype.is_floating_point_type:
            assert self.b_dtype.is_signed
            assert self.b_dtype.exponent_bits <= self.a_dtype.exponent_bits
            assert self.b_dtype.mantissa_bits <= self.a_dtype.mantissa_bits
            assert self.b_dtype.exponent_bits >= 1
        elif self.b_dtype.is_floating_point_type and not self.a_dtype.is_integer_type:
            # not implemented yet
            raise NotImplementedError

        if self.mma_config.use_f16_accum:
            if self.a_dtype == dtypes.float8e4m3:
                assert self.b_dtype.is_integer_type or self.b_dtype.exponent_bits <= 4
            else:
                assert self.a_dtype == dtypes.float16

    def check_config(self):
        # 16-bit activation don't support input scale
        # for 8bit/4-bit activation, we enable input scale by default
        if self.a_dtype.num_bits == 16:
            assert self.quant_param_config.has_input_scale is not True
        if self.quant_param_config.has_input_scale is None:
            self.quant_param_config.has_input_scale = self.a_dtype.num_bits != 16

        if self.pipeline_config.use_warp_spec:
            assert self.pipeline_config.use_mbarrier
        if not self.quant_param_config.has_weight_scale:
            self.pipeline_config.use_tma_bs = False
        if not self.quant_param_config.has_zero_point:
            self.pipeline_config.use_tma_bzp = False
        if not self.epilogue_config.has_bias:
            self.pipeline_config.use_tma_bias = False

    def __call__(self):
        msg = (
            "don't call HummingKernel object directly, "
            "please use humming.ops.launch_kernel([kernel.kernel_id], ...) instead."
        )
        raise NotImplementedError(msg)
