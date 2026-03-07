import os
from typing import Callable

import torch
import torch.utils.cpp_extension

_libs = {}
_launcher_inited = False


def register_op(
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


def init_humming_launcher():
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
