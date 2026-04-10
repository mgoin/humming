import glob
import json
import os
import subprocess
from pathlib import Path

from cuda.bindings import nvrtc
from filelock import FileLock

import humming.jit.utils as jit_utils


class Compiler:
    @classmethod
    def signature(self):
        raise NotImplementedError

    @staticmethod
    def humming_include_dir():
        dirname = os.path.dirname(__file__)
        dirname = os.path.abspath(dirname + "/../include/")
        return dirname

    @staticmethod
    def include_dirs():
        return [Compiler.humming_include_dir()]

    @staticmethod
    def cuh_last_update_time():
        dirname = Compiler.humming_include_dir()
        data = {}
        for filename in sorted(glob.glob(f"{dirname}/**/*.cuh", recursive=True)):
            data[filename] = os.stat(filename).st_mtime
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def compile(cls, code, sm_version, kernel_expr):
        flags = cls.get_flags(sm_version)
        signature = f"{cls.__name__}$${cls.signature()}$${flags}$${kernel_expr}$${code}"
        signature += "$$" + Compiler.cuh_last_update_time()
        hash_hex = jit_utils.hash_to_hex(signature)

        cache_dirname = Path(os.path.join(jit_utils.get_humming_cache_dir(), hash_hex))
        cache_filename = cache_dirname / "kernel.cubin"
        cache_dirname.mkdir(exist_ok=True, parents=True)

        lock_filename = jit_utils.get_humming_lock_filename(hash_hex)
        with FileLock(lock_filename):
            if cache_filename.exists():
                return cache_filename.as_posix()

            cache_dirname.mkdir(exist_ok=True, parents=True)
            source_path = os.path.join(cache_dirname, "kernel.cu")
            with open(cache_dirname / "kernel.cu", "w") as f:
                f.write(code)
            with open(cache_dirname / "signature.txt", "w") as f:
                f.write(signature)

            compile_res = cls._compile(source_path, cache_dirname, sm_version, kernel_expr)
            returncode, stdout, stderr = compile_res

            with open(cache_dirname / "stdout.log", "w") as f:
                f.write(stdout)
            with open(cache_dirname / "stderr.log", "w") as f:
                f.write(stderr)

        if returncode != 0:
            print(stderr, flush=True)
            (cache_dirname / "kernel_tmp.cubin").unlink(missing_ok=True)
            raise RuntimeError(f"{cls} run failed")

        os.replace(cache_dirname / "kernel_tmp.cubin", cache_dirname / "kernel.cubin")
        return cache_filename.as_posix()

    @classmethod
    def get_flags(cls, sm_version):
        raise NotImplementedError

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr):
        raise NotImplementedError


class NVRTCCompiler(Compiler):
    _STD_HEADER_SHIMS: dict[str, str] = {
        "climits": "#include <cuda/std/climits>",
        "cfloat": "#include <cuda/std/cfloat>",
        "cstddef": """
            #include <cuda/std/cstddef>
            using namespace cuda::std;
        """,
        "cstdint": """
            #include <cuda/std/cstdint>
            #include <cuda/std/type_traits>
            using namespace cuda::std;
            namespace std {
            using cuda::std::is_same;
            using cuda::std::conditional_t;
            using cuda::std::conditional;
            using cuda::std::enable_if;
            using cuda::std::enable_if_t;
            }
        """,
        "type_traits": """
            #include <cuda/std/type_traits>
            namespace std {
            using cuda::std::is_same;
            using cuda::std::conditional_t;
            using cuda::std::conditional;
            using cuda::std::enable_if;
            using cuda::std::enable_if_t;
            }
        """,
        "cuda.h": """
            #pragma once
            #include <cuda/std/cstdint>
            using namespace cuda::std;
            typedef uint64_t cuuint64_t;
            #define CU_TENSOR_MAP_NUM_QWORDS 16
            typedef struct CUtensorMap_st {
                alignas(64) cuuint64_t opaque[CU_TENSOR_MAP_NUM_QWORDS];
            } CUtensorMap;
        """,
    }

    @classmethod
    def _get_std_header_shims(cls):
        names = []
        sources = []

        for header, content in cls._STD_HEADER_SHIMS.items():
            names.append(header.encode())
            sources.append(content.encode())
        return names, sources

    @classmethod
    def signature(cls):
        _, major, minor = nvrtc.nvrtcVersion()
        return f"nvrtc+{major}.{minor}"

    @classmethod
    def get_flags(cls, sm_version):
        flags = [
            f"--gpu-architecture=sm_{sm_version}",
            "-std=c++17",
            "--use_fast_math",
            "--dopt=on",
            "-extra-device-vectorization",
            "--ptxas-options=-O3",
            "--ptxas-options=--register-usage-level=10",
            "--diag-suppress=39,161,174,177,940",
            "-default-device",
        ]
        for d in cls._get_include_dirs():
            flags.append(f"-I{d}")
        return flags

    @classmethod
    def _get_include_dirs(cls):
        dirs = list(cls.include_dirs())
        cuda_include = jit_utils.get_cuda_include_path()
        if os.path.isdir(cuda_include):
            dirs.append(cuda_include)

        try:
            import nvidia

            for pkg_path in nvidia.__path__:
                for sub in os.listdir(pkg_path):
                    inc = os.path.join(pkg_path, sub, "include")
                    if os.path.isdir(inc):
                        dirs.append(inc)
        except (ImportError, OSError):
            pass
        return dirs

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr):
        from cuda.bindings import nvrtc

        with open(source_path, "r") as f:
            code = f.read()

        flags = cls.get_flags(sm_version)

        shim_names, shim_sources = cls._get_std_header_shims()
        err, prog = nvrtc.nvrtcCreateProgram(
            code.encode(),
            b"kernel.cu",
            len(shim_names),
            shim_sources,
            shim_names,
        )
        assert err == nvrtc.nvrtcResult.NVRTC_SUCCESS, f"nvrtcCreateProgram failed: {err}"

        if kernel_expr:
            name_expr = " ".join(kernel_expr.split())
            nvrtc.nvrtcAddNameExpression(prog, name_expr.encode())

        opts = [f.encode() for f in flags]
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]

        _, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b"\0" * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        stderr = log.decode(errors="replace").rstrip("\0")

        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            nvrtc.nvrtcDestroyProgram(prog)
            return 1, "", stderr

        _, cubin_size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b"\0" * cubin_size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        nvrtc.nvrtcDestroyProgram(prog)

        target_path = cache_dirname / "kernel_tmp.cubin"
        with open(target_path, "wb") as f:
            f.write(cubin)

        with open(cache_dirname / "cmdline.json", "w") as f:
            json.dump({"compiler": "nvrtc", "flags": flags}, f, ensure_ascii=False)

        return 0, "", stderr


class NVCCCompiler(Compiler):
    @classmethod
    def signature(cls):
        nvcc_path = jit_utils.get_cuda_command_path("nvcc")
        nvcc_version = jit_utils.get_cuda_nvcc_version(nvcc_path)
        return "nvcc+" + nvcc_version

    @classmethod
    def get_flags(cls, sm_version):
        cxx_flags = [
            "-fPIC",
            "-O3",
            "-Wno-deprecated-declarations",
            "-Wno-abi",
            "-fopenmp",
            "-lgomp",
        ]

        return [
            "-std=c++17",
            "--ptxas-options=--register-usage-level=10",
            "--use_fast_math",
            "--diag-suppress=39,161,174,177,940,177",
            *[f"-I{d}" for d in cls.include_dirs()],
            f"-gencode=arch=compute_{sm_version},code=sm_{sm_version}",
            "-cubin",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            f"--compiler-options={','.join(cxx_flags)}",
        ]

    @classmethod
    def _compile(cls, source_path, cache_dirname, sm_version, kernel_expr):
        if kernel_expr:
            with open(source_path, "a") as f:
                f.write(f"\nauto ptr = reinterpret_cast<void*>(&{kernel_expr});\n")

        nvcc_path = jit_utils.get_cuda_command_path("nvcc")
        target_path = (cache_dirname / "kernel_tmp.cubin").as_posix()

        cmd = [nvcc_path, source_path, "-o", target_path] + cls.get_flags(sm_version)
        with open(cache_dirname / "cmdline.json", "w") as f:
            json.dump(cmd, f, ensure_ascii=False)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode, result.stdout, result.stderr
