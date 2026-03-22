import argparse

import torch
import triton
from tqdm import tqdm

from humming import dtypes, ops
from humming.layer import HummingLayer
from humming.utils.test import (
    generate_random_moe_tensors,
    random_fill_tensor,
    save_benchmark_result,
)


def bench_humming(
    shape_n: int,
    shape_k: int,
    a_dtype: str,
    b_dtype: str,
    c_dtype: str,
    bs_dtype: str,
    input_scale_group_size: int = 0,
    weight_scale_group_size: int = 0,
    num_experts: int | None = None,
    top_k: int = 0,
    has_zero_point: bool = False,
    is_fp_zero_point: bool = False,
    use_f16_accum: bool = False,
    is_moe_down: bool = False,
    shape_m_list: list[int] | None = None,
) -> list[dict[str, int | float]]:
    torch_dtype = dtypes.torch_dtype_map[dtypes.DataType.from_str(c_dtype)]
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        top_k=top_k,
        weight_config={
            "dtype": b_dtype,
            "group_size": weight_scale_group_size,
            "scale_dtype": bs_dtype,
            "has_zero_point": has_zero_point,
            "is_fp_zero_point": is_fp_zero_point,
        },
        input_config={"dtype": a_dtype, "group_size": input_scale_group_size},
        torch_dtype=torch_dtype,
        is_moe_down=is_moe_down,
        activation_type=("silu" if num_experts is not None and not is_moe_down else "none"),
    ).to("cuda:0")

    for tensor in layer.parameters():
        random_fill_tensor(tensor)
    layer.transform()
    layer.prepare_default_kernel_configs(use_f16_accum=use_f16_accum)

    default_shape_m_list = [2**i for i in range(13)]
    benchmark_result: list[dict[str, int | float]] = []
    for shape_m in tqdm(shape_m_list or default_shape_m_list):
        if num_experts is not None and is_moe_down:
            inputs = torch.randn((shape_m, top_k, shape_k), dtype=torch_dtype, device="cuda:0")
        else:
            inputs = torch.randn((shape_m, shape_k), dtype=torch_dtype, device="cuda:0")

        input_scale: torch.Tensor | None = None
        if a_dtype not in ["float16", "bfloat16"]:
            inputs, input_scale = ops.quant_input(inputs, a_dtype, None, input_scale_group_size)

        if num_experts is None:
            topk_weights: torch.Tensor | None = None
            sorted_token_ids: torch.Tensor | None = None
            expert_ids: torch.Tensor | None = None
            num_tokens_padded: torch.Tensor | None = None
        else:
            torch.cuda.manual_seed(shape_m)
            moe_tensors = generate_random_moe_tensors(
                shape_m=shape_m,
                num_experts=num_experts,
                top_k=top_k,
                block_size_config=layer.humming_block_size_configs[""],
            )
            _, topk_weights, sorted_token_ids, expert_ids, num_tokens_padded = moe_tensors

        def run():
            return layer(
                inputs=inputs,  # noqa
                input_scale=input_scale,  # noqa
                topk_weights=topk_weights,  # noqa
                sorted_token_ids=sorted_token_ids,  # noqa
                expert_ids=expert_ids,  # noqa
                num_tokens_padded=num_tokens_padded,  # noqa
            )

        outputs = run()
        torch.cuda.synchronize()
        t = triton.testing.do_bench(run, warmup=100, rep=1000)

        nbytes = inputs.nbytes + outputs.nbytes
        if input_scale is not None:
            nbytes += input_scale.nbytes
        for tensor in layer.state_dict().values():
            if num_experts is None:
                nbytes += tensor.nbytes
            else:
                assert expert_ids is not None
                num_actived_experts = len(set(expert_ids.tolist()))
                nbytes += tensor.nbytes // num_experts * num_actived_experts

        res = {
            "shape_m": shape_m,
            "time": t,
            "memory_gbps": nbytes / t / 1e6,
            "compute_tops": shape_m * shape_n * shape_k * (top_k or 1) * 2 / t / 1e9,
        }
        benchmark_result.append(res)

    return benchmark_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape_n", type=int, required=True)
    parser.add_argument("--shape_k", type=int, required=True)
    activation_dtypes = [
        "float16",
        "bfloat16",
        "float8e4m3",
        "float8e5m2",
        "float4e2m1",
        "int8",
        "int4",
    ]
    scale_dtypes = ["float16", "bfloat16", "float8e4m3", "float8e5m2", "float8e8m0"]
    f16_dtypes = ["float16", "bfloat16"]
    parser.add_argument("--a_dtype", type=str, choices=activation_dtypes, required=True)
    parser.add_argument("--b_dtype", type=str, required=True)
    parser.add_argument("--bs_dtype", type=str, choices=scale_dtypes, required=True)
    parser.add_argument("--c_dtype", type=str, choices=f16_dtypes, required=True)
    parser.add_argument("--input_scale_group_size", type=int, default=0)
    parser.add_argument("--weight_scale_group_size", type=int, default=0)
    parser.add_argument("--zero_point", default=False, action="store_true")
    parser.add_argument("--use_fp_zero_point", default=False, action="store_true")
    parser.add_argument("--use_f16_accum", default=False, action="store_true")
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--is_moe_down", default=False, action="store_true")
    parser.add_argument("--shape_m_list", type=int, default=None, nargs="+")
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()
    benchmark_result = bench_humming(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        a_dtype=args.a_dtype,
        b_dtype=args.b_dtype,
        c_dtype=args.c_dtype,
        bs_dtype=args.bs_dtype,
        num_experts=args.num_experts,
        top_k=args.top_k,
        input_scale_group_size=args.input_scale_group_size,
        weight_scale_group_size=args.weight_scale_group_size,
        has_zero_point=args.zero_point or args.use_fp_zero_point,
        is_fp_zero_point=args.use_fp_zero_point,
        use_f16_accum=args.use_f16_accum,
        shape_m_list=args.shape_m_list,
        is_moe_down=args.is_moe_down,
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
