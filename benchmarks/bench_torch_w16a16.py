import argparse

import torch
import triton
from tqdm import tqdm

from humming.utils.test import save_benchmark_result


def bench_torch_w16a16(
    shape_n: int,
    shape_k: int,
    dtype: str,
    use_f16_accum: bool = False,
    shape_m_list: list[int] | None = None,
) -> list[dict[str, int | float]]:
    assert dtype in ["float16", "bfloat16"]
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if use_f16_accum:
        assert torch_dtype == torch.float16
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

    weight = torch.randn((shape_n, shape_k), dtype=torch_dtype, device="cuda:0")

    default_shape_m_list = [2**i for i in range(15)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        inputs = torch.randn((shape_m, shape_k), dtype=torch_dtype, device="cuda:0")

        weight_t = weight.T

        def run():
            return inputs.matmul(weight_t)  # noqa

        outputs = run()
        torch.cuda.synchronize()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        nbytes = inputs.nbytes + outputs.nbytes + weight.nbytes
        res = {
            "shape_m": shape_m,
            "time": t,
            "memory_gbps": nbytes / t / 1e6,
            "compute_tops": shape_m * shape_n * shape_k * 2 / t / 1e9,
        }
        benchmark_result.append(res)

    return benchmark_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    f16_dtypes = ["float16", "bfloat16"]
    parser.add_argument("--dtype", type=str, choices=f16_dtypes, required=True)
    parser.add_argument("--use_f16_accum", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_torch_w16a16(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        dtype=args.dtype,
        use_f16_accum=args.use_f16_accum,
        shape_m_list=args.shape_m_list,
    )

    save_benchmark_result(benchmark_result, args)

    from tabulate import tabulate

    table = tabulate(
        benchmark_result,
        headers="keys",
        tablefmt="grid",
        numalign="right",
        floatfmt=".4f",
    )

    print(table)


if __name__ == "__main__":
    main()
