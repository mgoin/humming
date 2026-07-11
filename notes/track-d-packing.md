# Track d-packing findings log

Goal: tcgen05 TS-mode weight/scale/zp packing against the register-layout
CONTRACT (jinzhen-umma-notes.md §4), as the reference we hand Jinzhen.

## Same-GPU SS-mode baseline (GPU 3, B300 sm_103a, 2026-07-11)

`benchmarks/bench_baseline_track_d.py` (mma.sync vs shipped SS-mode tcgen05,
production config M128/N128/K128 s4 WS; M64 for M=16):

```
shape                M  mma.sync us    tcg-ss us              cfg  tcg/mma
Llama70B gate       16        166.4        230.7     M64K128s4+ws    0.72x
Llama70B gate      128        329.0        232.1    M128K128s4+ws    1.42x
Llama70B gate      512       1062.1        786.1    M128K128s4+ws    1.35x
Llama70B gate     2048       3992.7       2840.1    M128K128s4+ws    1.41x
Llama70B down       16        280.8        393.7     M64K128s4+ws    0.71x
Llama70B down      128        280.8        398.1    M128K128s4+ws    0.71x
Llama70B down      512       1124.5        798.8    M128K128s4+ws    1.41x
Llama70B down     2048       3875.0       2783.3    M128K128s4+ws    1.39x
```

Matches workbook B.35 expectations (1.35-1.42x at M>=512). Track d is
packing-only: no kernel-perf change expected; this table is the "did we
break anything" reference.

## Facts established (with evidence)

1. **The lop3 (i, i+4) interleave, precisely**: humming's `uint_to_f16`
   dequants a 32-bit word into 4 regs; reg `i` = (value-slot `i` in lo
   half, value-slot `i + V/2` in hi half), V = 32/kBits values per word
   (u4: V=8, slots i and i+4 -- the "(i, i+4) interleave"). Pack-time
   compensation (`get_interleaved_index` in process.cuh): value-slot `s`
   of a word holds logical element `e = (s % (V/2))*2 + s/(V/2)`;
   inversely element `e` lands in slot `s = (e%2)*(V/2) + e/2`. With
   logical element order = ascending K per row, dequant reg r comes out
   as the (K=2r, K=2r+1) bf16 pair -- exactly what TS-mode TMEM A needs.
   Same formula covers u8 (V=4). Verified two independent ways:
   - `test_ts_register_contract_simulation` (Python loader+dequant sim).
   - `test_mma_sync_inverse_matches_cuda_repack`: my Python model of the
     EXISTING mma.sync layout (same slot formula, fragment element order)
     is bit-exact against CUDA `ops.repack_weight` at 4 shapes.

2. **The existing mma.sync layout, decoded** (needed as the base):
   packed[c = k/16, col = 128*(n/64) + 4*tid + (n%64)/16] bits
   [4s, 4s+4) with tid = 4*(n%8) + (k%8)/2,
   e' = 4*((n%16)/8) + 2*((k%16)/8) + k%2, s = (e'%2)*4 + e'/2.
   Note col mixes n AND k (thread owns an m16n8-fragment slice); in the
   TS layout col depends only on (n, (k%16)/8) -- thread owns a row.

3. **TS layout can keep the file SHAPE and g2s path 100% unchanged**:
   same int32 [K/16, 2N] tensor (u4), same 64-row x 16-K block tiling,
   same TMA/seek()/kSmemStride math. Only the permutation inside each
   128-word block changes. loader_b's WarpN==32 bf16 half-group path
   (kIsWarpHalfGroup) then delivers each thread's 2 words (u4) with NO
   loader change: int2 index = 2*(32*(w/2) + lane) + w%2 -> 16-B slot
   `lane` = [row 64*(w/2)+lane | row 64*(w/2)+32+lane] halves.
   warp_id%2 == n_warp_id%2 holds because N_WARPS = BlockN/32 is even.

4. **u4/bf16 never takes the int2fp/zp preprocessing path**
   (`should_preprocess_for_int2fp` false for b_bits <= 6 with zp, <= 7
   without) -- so the W4A16 TS weight repack is value-preserving and
   gs only affects the scale/zp streams.

## Dead ends / gotchas

* First attempt at describing the TS slot order tried to reuse the
  mma.sync `tmp[]` index math with only i-dims permuted -- doesn't work:
  the repack thread's GATHER rows change (rows {tid, tid+32} instead of
  {tid/4 + 8m}), not just the element order. The CUDA variant needs its
  own gather loop, not just a different tmp permutation.
* `nvidia-smi` ignores CUDA_VISIBLE_DEVICES -- always export it and let
  torch pick device 0 (= physical GPU 3).

## Status

- [x] Baseline recorded (above).
- [x] Python reference packer + inverse + scale/zp streams
      (`humming/utils/ts_packing.py`), round-trip + contract-sim tests
      green (19 CPU tests + 4 GPU cross-validation tests).
- [ ] CUDA repack kernel variant (`use_tcgen05_ts`) bit-exact vs ref.
- [ ] docs/tcgen05_ts_packing.md spec + Jinzhen interface contract.
- [ ] Cross-check vs track b's packer if landed.
