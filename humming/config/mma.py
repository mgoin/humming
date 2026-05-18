import math
import re

import humming.dtypes as dtypes
from humming.config.enum import MmaType

DTYPE_BIT_WIDTH_MAP = {
    "f32": 32,
    "s32": 32,
    "f16": 16,
    "bf16": 16,
    "e4m3": 8,
    "e5m2": 8,
    "s8": 8,
    "e2m1": 4,
    "s4": 4,
}

DTYPE_MAP = {
    dtypes.float32: "f32",
    dtypes.int32: "s32",
    dtypes.float16: "f16",
    dtypes.bfloat16: "bf16",
    dtypes.float8e4m3: "e4m3",
    dtypes.float8e5m2: "e5m2",
    dtypes.int8: "s8",
    dtypes.float4e2m1: "e2m1",
    dtypes.int4: "s4",
}


def calc_reg_count(rows, cols, ptx_dtype):
    total_bits = rows * cols * DTYPE_BIT_WIDTH_MAP[ptx_dtype]
    assert total_bits % (32 * 32) == 0
    reg_count = total_bits // (32 * 32)
    return reg_count


class MmaOpClassImpl:
    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        self.reg_a_count = calc_reg_count(m, k, self.a_dtype)
        self.reg_b_count = calc_reg_count(k, n, self.b_dtype)
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype)
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::MMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            "",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint32_t *a, uint32_t *b, {reg_cd_type} *c, {reg_cd_type} *d) {{",
            *self.generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=0):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        shape = self.shape

        asm_op = f"mma.sync.aligned.m{shape[0]}n{shape[1]}k{shape[2]}.row.col"
        asm_op += f".{cd_dtype}.{a_dtype}.{b_dtype}.{cd_dtype}"
        if "s" in a_dtype:
            asm_op += ".satfinite"

        start = 0
        end = 0
        param_placeholders_list = []
        counts = [self.reg_cd_count, self.reg_a_count, self.reg_b_count, self.reg_cd_count]
        for i in range(len(counts)):
            end += counts[i]
            placeholder_str = ", ".join(f"%{x}" for x in range(start, end))
            param_placeholders_list.append("{" + placeholder_str + "}")
            start += counts[i]

        a_params = []
        b_params = []
        c_params = []
        d_params = []
        for i in range(self.reg_a_count):
            a_params.append(f' "r"(a[{i}])')
        for i in range(self.reg_b_count):
            b_params.append(f' "r"(b[{i}])')
        for i in range(self.reg_cd_count):
            t = "f" if cd_dtype == "f32" else "r"
            c_params.append(f' "{t}"(c[{i}])')
            d_params.append(f'"+{t}"(d[{i}])')

        asm_code = f"""
        asm volatile(
          "{asm_op} "
          "{", ".join(param_placeholders_list)};\\n"
          : {", ".join(d_params)}
          : {", ".join(a_params)},
            {", ".join(b_params)},
            {", ".join(c_params)}
        );
        """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class WgmmaOpClassImpl:
    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        self.reg_a_count = calc_reg_count(m, k, self.a_dtype) // 4
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype) // 4
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::WGMMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            "",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint32_t *a, uint64_t &desc, {reg_cd_type} *d, bool pred = true) {{",
            *self.generate_ptx(indent=2, has_scale_d=True).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=2, has_scale_d=True):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        shape = self.shape

        asm_op = f"wgmma.mma_async.sync.aligned.m{shape[0]}n{shape[1]}k{shape[2]}"
        asm_op += f".{cd_dtype}.{a_dtype}.{b_dtype}"
        if "s" in a_dtype:
            asm_op += ".satfinite"

        start = 0
        end = 0
        param_placeholders_list = []
        counts = [self.reg_cd_count, self.reg_a_count]
        for i in range(len(counts)):
            end += counts[i]
            placeholder_str = ", ".join(f"%{x}" for x in range(start, end))
            param_placeholders_list.append("{" + placeholder_str + "}")
            start += counts[i]
        param_placeholders_list.append(f"%{sum(counts)}")

        other_ptx_args = ", p" if has_scale_d else ", 1"
        if self.a_dtype in ["f16", "bf16"]:
            other_ptx_args += ", 1, 1, 0"
        elif self.a_dtype in ["e4m3", "e5m2", "e2m1"]:
            other_ptx_args += ", 1, 1"

        b_param = ' "l"(desc)'
        a_params = []
        cd_params = []
        for i in range(self.reg_a_count):
            a_params.append(f' "r"(a[{i}])')
        for i in range(self.reg_cd_count):
            t = "f" if cd_dtype == "f32" else "r"
            cd_params.append(f'"+{t}"(d[{i}])')

        cd_param_str = ""
        for i in range(math.ceil(len(cd_params) / 4)):
            cd_params_part = cd_params[i * 4 : (i + 1) * 4]
            cd_params_part_str = ", ".join(cd_params_part) + ",\n"
            if cd_param_str:
                cd_params_part_str = "    " + cd_params_part_str

            cd_param_str += cd_params_part_str

        cd_param_str = cd_param_str.strip().strip(",")

        if has_scale_d:
            asm_code = f"""
            asm volatile(
              "{{\\n"
                ".reg .pred p;\\n"
                "setp.ne.b32 p, %{sum(counts) + 1}, 0;\\n"
                "{asm_op} "
                "{", ".join(param_placeholders_list)}{other_ptx_args};\\n"
              "}}\\n"
              : {cd_param_str}
              : {", ".join(a_params)},
                {b_param}, "r"((uint32_t)pred)
            );
            """
        else:
            asm_code = f"""
            asm volatile(
            "{asm_op} "
            "{", ".join(param_placeholders_list)}{other_ptx_args};\\n"
            : {cd_param_str}
            : {", ".join(a_params)},
                {b_param}
            );
            """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class Tcgen05OpClassImpl:
    """Blackwell tcgen05.mma (UMMA) instruction descriptor.

    Differences from MmaOpClassImpl / WgmmaOpClassImpl:
      * Both operands are read from SMEM (descriptors), so there are no
        A/B register-tile counts to emit.
      * Accumulator lives in TMEM (column allocation), not RMEM. The
        epilogue does its own t2r load, so we still emit a CRegisters
        alias for the RMEM-side tile shape, but the in-mainloop fma() does
        not touch C registers directly.

    The actual PTX `tcgen05.mma` emission lives in
    `include/humming/utils/ptx/tcgen05.cuh` and is invoked by
    `mma/tcgen05_mma.cuh` rather than inlined here, because the
    instruction takes a runtime instruction-descriptor that must be
    constructed at kernel time (not at codegen time).
    """

    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype,
                 warp_shape=None):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        # tcgen05.mma writes a (BlockM × BlockN) accumulator into TMEM. The
        # epilogue's t2r reads it back into RMEM, distributed across the
        # epilogue warps (after K-reduction). Per-thread CRegisters count:
        #   warp_M * warp_N * cd_bits / (32 lanes * 32 bits/uint32)
        # Falls back to whole-tile sizing if warp_shape isn't provided
        # (used by callers that only have access to MMA shape).
        if warp_shape is not None:
            warp_m, warp_n = warp_shape[0], warp_shape[1]
            cd_bits = DTYPE_BIT_WIDTH_MAP[self.cd_dtype]
            self.reg_cd_count = warp_m * warp_n * cd_bits // (32 * 32)
        else:
            self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype) // 4
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype for tcgen05: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        lines = [
            "static constexpr MmaType kMmaType = MmaType::TCGEN05;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            "",
            # tcgen05.mma operands come from SMEM/TMEM descriptors, not
            # per-warp register tiles. The TCGEN05 class in
            # mma/tcgen05_mma.cuh constructs descriptors at runtime; we
            # only need to expose the C-register footprint here so the
            # epilogue path can size its t2r staging.
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "// fma() not emitted here -- see TCGEN05::run() in",
            "// mma/tcgen05_mma.cuh for the tcgen05.mma issue path.",
        ]
        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"
        return code


class MmaOpClass:
    @classmethod
    def from_config(cls, mma_type, m, n, k, a_dtype, b_dtype, cd_dtype,
                    warp_shape=None):
        mma_type = mma_type if isinstance(mma_type, MmaType) else getattr(MmaType, mma_type.upper())

        if mma_type == MmaType.MMA:
            return MmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype)
        elif mma_type == MmaType.WGMMA:
            return WgmmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype)
        elif mma_type == MmaType.TCGEN05:
            return Tcgen05OpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype,
                                      warp_shape=warp_shape)
        else:
            raise ValueError(f"Invalid MMA Type: {mma_type}")
