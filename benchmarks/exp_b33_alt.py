"""Part-1 experiment: per-TILE TMEM accumulator rotation (tcgen05_acc_stages).

Workbook B.33 showed per-K-ITER column alternation loses; this is the
different per-tile variant: tile i's epilogue drain overlaps tile i+1's
producer loads. Controlled by `tcgen05_acc_stages` (1 = shipped behaviour,
2 = rotate two BlockN-wide TMEM accumulator buffers).

acc_stages=2 moves `smem.reduce` out of the stage union (+32 KB at
BM=BN=128), which does NOT fit next to 4 BK=128 stages -> the s4 PROD
config fails cuFuncSetAttribute. Compare at s3 (both settings), with
acc1-s4 shipped numbers as the reference row.

Regimes:
  * tile-count >> SM-count: Llama70B down at M=2048/4096.
  * epilogue-exposed small-K: 4096x4096 at M=2048 (multi-wave) and M=256
    (single wave -- rotation can win nothing there, included honestly).

Run: CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/exp_b33_alt.py
"""

import sys
import time

import torch

sys.path.insert(0, "tests")
from test_tcgen05 import _build_w4a16_problem  # noqa: E402

from humming import dtypes, ops  # noqa: E402
from humming.kernel.humming import HummingKernel  # noqa: E402


def build(shape_m, shape_n, shape_k, acc_stages, num_stages,
          with_ref=False):
    torch.manual_seed(123)
    inputs_ref, inputs, weight, weight_scale, zero_point, weight_ref = \
        _build_w4a16_problem(shape_m, shape_n, shape_k, 128, True)
    kernel = HummingKernel(
        shape_n=shape_n, shape_k=shape_k,
        block_shape=(128, 128, 128), warp_shape=(32, 64, 128),
        a_dtype=dtypes.bfloat16, b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16, bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128, has_zero_point=True,
        num_stages=num_stages, use_warp_spec=True, use_tma=True,
        use_cp_async=False, use_mbarrier=True, use_tma_bzp=False,
        has_bias=False, mma_type="tcgen05", use_tcgen05=True,
        use_stream_k=False, tcgen05_acc_stages=acc_stages,
    )
    outputs = torch.empty((shape_m, shape_n), dtype=torch.bfloat16,
                          device="cuda")

    def launch():
        ops.launch_kernel(
            configs=[kernel.kernel_id], inputs=inputs, weight=weight,
            outputs=outputs, weight_scale=weight_scale,
            zero_point=zero_point,
        )

    ref = None
    if with_ref:
        ref = inputs_ref.matmul(weight_ref.T).to(torch.bfloat16)
    return launch, outputs, ref


def time_kernel(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def check(shape_m, shape_n, shape_k, acc_stages, num_stages=3):
    launch, out, ref = build(shape_m, shape_n, shape_k, acc_stages,
                             num_stages, with_ref=True)
    launch()
    torch.cuda.synchronize()
    err = (out.float() - ref.float()).abs()
    try:
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=0.5)
        ok = True
    except AssertionError:
        ok = False
    print(f"correctness s{acc_stages} stages={num_stages} "
          f"{shape_m}x{shape_n}x{shape_k}: max|err|={err.max().item():.3e} "
          f"mean|err|={err.mean().item():.3e} ok={ok}", flush=True)
    return ok


def main():
    # Shapes: multi-tile-per-CTA (rotation exercised), tail (odd-tile
    # final-pending drain), single-tile-per-CTA (M=128: pending drain at
    # loop exit ONLY -- this is the shape the predecessor's hung test
    # used), single wave.
    all_ok = True
    for m, n, k in [(128, 128, 4096), (2048, 4096, 4096),
                    (2112, 4096, 4096), (256, 4096, 4096)]:
        for s in (1, 2):
            all_ok &= check(m, n, k, s)
    if not all_ok:
        print("CORRECTNESS FAILED -- perf numbers below are meaningless")

    for label, m, n, k in [
        ("Llama70B down", 2048, 8192, 28672),
        ("Llama70B down", 4096, 8192, 28672),
        ("4096x4096", 2048, 4096, 4096),
        ("4096x4096", 256, 4096, 4096),
    ]:
        row = f"{label:<14s} M={m:<5d}"
        for acc, stages in [(1, 4), (1, 3), (2, 3)]:
            launch, _, _ = build(m, n, k, acc, stages)
            t = time_kernel(launch)
            row += f"  a{acc}s{stages}={t:8.1f}us"
        print(row, flush=True)


if __name__ == "__main__":
    main()
