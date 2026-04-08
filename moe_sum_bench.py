

import torch
import triton

from humming import ops


def benchmark(M, N, K):
    A = torch.randn((M, N, K), device="cuda", dtype=torch.float16)
    B = torch.randn((M, N), device="cuda", dtype=torch.float16)

    def pytorch_naive(A, B):
        return (A * B.unsqueeze(-1)).sum(dim=1)

    def pytorch_sum_only(A):
        return torch.sum(A, dim=1)

    # 验证正确性
    out_pt = pytorch_naive(A, B)
    out_tr = ops.fused_moe_mul_sum(A, B)
    print("最大误差 Max diff:", (out_pt - out_tr).abs().max().item())

    # 测试性能
    ms_naive = triton.testing.do_bench(lambda: pytorch_naive(A, B))
    ms_sum_only = triton.testing.do_bench(lambda: pytorch_sum_only(A))
    ms_triton = triton.testing.do_bench(lambda: ops.fused_moe_mul_sum(A, B))

    print(f"{M} {N} {K} PyTorch 乘法+规约:  {ms_naive:.3f} ms")
    print(f"{M} {N} {K} PyTorch 仅仅执行 sum(A, dim=1): {ms_sum_only:.3f} ms")
    print(f"{M} {N} {K} Triton Fused Kernel: {ms_triton:.3f} ms")


if __name__ == "__main__":
    for N in [2, 4, 5, 8]:
        for K in [1024, 1536, 2048, 8192, 7168, 3072]:
            for M in [1, 6, 23, 86, 458, 2348, 4772, 8574]:
                benchmark(M, N, K)
