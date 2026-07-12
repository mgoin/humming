"""expand/moe-blockm32 M4: does BlockM=32 win at fine-grained MoE?

Times three grouped configs at the SAME shape/E point:
  * mma.sync grouped (BlockM=64, the wave-1 non-TS baseline)
  * TS BlockM=64  (the wave-1 TS fine-grained baseline)
  * TS BlockM=32  (the new small tile)

Correctness of each point is checked vs the per-expert bf16-rounded
dequant reference before timing. Timing uses CUDA events; each config is
timed REPEATS times and we report the median and min..max spread so a
noisy sample does not masquerade as a win.

Run on a VERIFIED-IDLE GPU (not GPU 0/1):
  CUDA_VISIBLE_DEVICES=<idle> .venv/bin/python benchmarks/bench_ts_moe_blockm32.py
"""
import statistics
import sys

import torch

from humming import dtypes, ops
from humming.kernel.humming import HummingKernel
from humming.utils.test import (
    generate_random_inputs,
    generate_random_moe_tensors,
    generate_random_weight,
)
from humming.utils.weight import (
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
)

A_DTYPE = dtypes.bfloat16
B_DTYPE = dtypes.uint4
C_DTYPE = dtypes.bfloat16
BS_DTYPE = dtypes.bfloat16
GROUP_SIZE = 128
TORCH_DTYPE = dtypes.torch_dtype_map[C_DTYPE]

# (label, N, K, [(num_experts, top_k), ...]); coarse E=8 is the control.
SHAPES = [
    ("Qwen3-MoE gate/up", 1536, 4096, [(8, 2), (128, 8)]),
    ("Qwen3-MoE down", 4096, 1536, [(8, 2), (128, 8)]),
    ("DeepSeek gate/up", 2048, 7168, [(8, 2), (256, 8)]),
    ("DeepSeek down", 7168, 2048, [(8, 2), (256, 8)]),
]
NUM_TOKENS = 512
REPEATS = 7
ITERS = 50
WARMUP = 15


def time_kernel(launch_fn):
    for _ in range(WARMUP):
        launch_fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            launch_fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / ITERS * 1e3)  # us
    return samples


def _make_layout(gemm_type, num_experts, m, top_k):
    avg = (m * top_k + num_experts - 1) // num_experts
    emt = None if gemm_type == "grouped_contiguous" else max(64, 4 * avg)
    for _ in range(6):
        try:
            _, layout, *_ = generate_random_moe_tensors(
                m, num_experts=num_experts, top_k=top_k,
                gemm_type=gemm_type, expert_max_tokens=emt)
            return layout, emt
        except AssertionError:
            emt *= 2
    raise RuntimeError("emt sizing failed")


def _build(gemm_type, N, K, num_experts, top_k, m, mode):
    """mode in {'mma', 'ts64', 'ts32'}. Returns (launch, check_fn)."""
    torch.manual_seed(0)
    layout, emt = _make_layout(gemm_type, num_experts, m, top_k)
    _, wref, wc, ws, zp, _ = generate_random_weight(
        n=N, k=K, group_size=GROUP_SIZE, dtype=B_DTYPE, scale_dtype=BS_DTYPE,
        num_experts=num_experts, has_zero_point=True)
    m_new = m * top_k if gemm_type == "grouped_contiguous" else num_experts * emt
    _, iref, inputs, _ = generate_random_inputs(
        m=m_new, k=K, group_size=0, dtype=A_DTYPE)

    use_ts = mode != "mma"
    weight = prepare_humming_weight(
        wc, B_DTYPE, A_DTYPE, zero_point=zp, use_tcgen05_ts=use_ts)

    if use_ts:
        bm = 32 if mode == "ts32" else 64
        wsp = prepare_humming_weight_scale(
            ws, to_apply_on_c=False, use_tcgen05_ts=True)
        zpp = prepare_humming_zero_point(
            zp, B_DTYPE, packed=False, use_tcgen05_ts=True)
        kern = HummingKernel(
            shape_n=N, shape_k=K, block_shape=(bm, 128, 64),
            warp_shape=(bm, 32, 64), a_dtype=A_DTYPE, b_dtype=B_DTYPE,
            c_dtype=C_DTYPE, bs_dtype=BS_DTYPE, num_experts=num_experts,
            num_stages=4, use_warp_spec=True, has_bias=False,
            has_zero_point=True, weight_scale_group_size=GROUP_SIZE,
            mma_type="tcgen05", use_tcgen05=True, use_tcgen05_ts=True,
            use_tma=True, use_mbarrier=True, use_tma_bzp=False,
            use_stream_k=False, gemm_type=gemm_type)
        assert kern.use_tcgen05_ts is True
    else:
        wsp = prepare_humming_weight_scale(ws, to_apply_on_c=False)
        zpp = prepare_humming_zero_point(zp, B_DTYPE, packed=False)
        kern = HummingKernel(
            shape_n=N, shape_k=K,
            block_shape=(64, 256, 32), warp_shape=(64, 64, 32),
            a_dtype=A_DTYPE, b_dtype=B_DTYPE, c_dtype=C_DTYPE, bs_dtype=BS_DTYPE,
            num_experts=num_experts, num_stages=3, use_warp_spec=False,
            has_bias=False, has_zero_point=True,
            weight_scale_group_size=GROUP_SIZE, mma_type="mma",
            use_tma=False, use_stream_k=False, gemm_type=gemm_type)
        assert kern.use_tcgen05_ts is not True

    out = torch.zeros((m_new, N), dtype=TORCH_DTYPE, device=inputs.device)

    def launch():
        return ops.launch_kernel(
            configs=[kern.kernel_id], inputs=inputs, weight=weight, outputs=out,
            weight_scale=wsp, zero_point=zpp, expert_layout=layout)

    res = launch().view(-1, N)
    torch.cuda.synchronize()
    ref = torch.zeros_like(res)
    wr = wref.to(torch.bfloat16).float()
    for e in range(num_experts):
        if gemm_type == "grouped_contiguous":
            o1 = int(layout[e])
            o2 = m_new if e == num_experts - 1 else int(layout[e + 1])
        else:
            o1 = emt * e
            o2 = o1 + int(layout[e])
        if o2 > o1:
            ref[o1:o2] = iref[o1:o2].matmul(wr[e].T).to(TORCH_DTYPE)
    err = (res.float() - ref.float()).abs()
    ok = (torch.isfinite(res).all().item()
          and err.max().item() <= 0.5 + 1e-2 * ref.float().abs().max().item())
    return launch, ok, err.max().item()


def _fmt(samples):
    med = statistics.median(samples)
    return med, min(samples), max(samples)


def main():
    idx = torch.cuda.current_device()
    print(f"device[{idx}]: {torch.cuda.get_device_name(idx)}  "
          f"repeats={REPEATS} iters={ITERS}\n")
    hdr = (f"{'shape':<20}{'mode':<11}{'E':>4}{'tk':>3}{'m_new':>7}"
           f"{'mma':>9}{'ts64':>9}{'ts32':>9}"
           f"{'32/mma':>8}{'32/64':>8}{'64/mma':>8}{'ok':>4}")
    for gemm_type in ["grouped_contiguous", "grouped_masked"]:
        print(hdr)
        print("-" * len(hdr))
        for label, N, K, points in SHAPES:
            for ne, tk in points:
                m = NUM_TOKENS
                try:
                    lm, okm, _ = _build(gemm_type, N, K, ne, tk, m, "mma")
                    l64, ok64, _ = _build(gemm_type, N, K, ne, tk, m, "ts64")
                    l32, ok32, _ = _build(gemm_type, N, K, ne, tk, m, "ts32")
                    sm = _fmt(time_kernel(lm))
                    s64 = _fmt(time_kernel(l64))
                    s32 = _fmt(time_kernel(l32))
                    m_new = (m * tk if gemm_type == "grouped_contiguous"
                             else "masked")
                    ok = okm and ok64 and ok32
                    print(f"{label:<20}{gemm_type.split('_')[1]:<11}"
                          f"{ne:>4}{tk:>3}{str(m_new):>7}"
                          f"{sm[0]:>9.1f}{s64[0]:>9.1f}{s32[0]:>9.1f}"
                          f"{s32[0]/sm[0]:>8.2f}{s32[0]/s64[0]:>8.2f}"
                          f"{s64[0]/sm[0]:>8.2f}{'Y' if ok else 'N':>4}")
                    print(f"{'  spread(min..max)':<20}{'':11}{'':4}{'':3}{'':7}"
                          f"{sm[1]:>5.1f}-{sm[2]:<3.1f}"
                          f"{s64[1]:>5.1f}-{s64[2]:<3.1f}"
                          f"{s32[1]:>5.1f}-{s32[2]:<3.1f}")
                except Exception as e:
                    print(f"{label:<20}{gemm_type.split('_')[1]:<11}{ne:>4}"
                          f"{tk:>3}  ERR {str(e)[:44]}")
        print()


if __name__ == "__main__":
    main()
