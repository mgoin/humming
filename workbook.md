# sm100-tcgen05 workbook

Living scratchpad. **What's true now**, not the journey. Append when you learn
something the next session would otherwise rediscover; trim when it becomes
stale.

## Goal

Add a Blackwell `tcgen05.mma` (UMMA) path to humming for **bf16 × uint4
W4A16, AWQ-style (per-group bf16 scales + uint4 zero-points,
group_size=128)**. Beat humming's existing mma.sync path at compute-bound
batch sizes; coexist with it below the crossover.

## Environment

Branch `sm100-tcgen05` on `origin = github.com/mgoin/humming`. Editable
install at `/home/mgoin/code/vllm/humming/`; changes are live without
reinstall (`ops/utils.py` patched so `humming.__file__` works even when
cwd is the vllm root).

* Hardware: B300 SXM6, cc 10.3, compile target `sm_103a`.
* Toolchain: CUDA 13.0, `nvidia-cutlass-dsl==4.5.1`, torch in
  `~/code/vllm/.venv`. NVRTC for the device code; the JIT cache lives at
  `~/.humming/cache/<hash>/{kernel.cu, kernel.cubin, signature.txt}`.
* `rm -rf ~/.humming/cache` to force a fresh JIT after include-only edits.

## Reference perf bar (Sablefish on the same B300)

Sablefish (CUTLASS tcgen05 W4A16, AWQ int4 gs=128) is the upstream we have
to land at or below. `tests/baseline_phaseA.tsv` is the full table; the
headline:

| shape (N×K)     | B=16 hum vs sf | B=4096 hum vs sf |
| --------------- | -------------- | ---------------- |
| 4096 × 4096     |  16 / 41 µs    | 343 / 228 µs     |
| 14336 × 4096    |  17 / 62 µs    | 1163 / 651 µs    |
| 28672 × 8192    |  58 / 194 µs   | 4536 / 2363 µs   |
| 8192 × 28672    |  60 / 181 µs   | 4486 / 2289 µs   |

Humming today **wins at low batch** (mma.sync's small fragments are
inherently better for skinny GEMM) and roughly matches Marlin at high
batch but loses to Sablefish by ~2×. The tcgen05 path is meant to close
that high-batch gap.

## Where tcgen05 fits in humming

`MmaType.TCGEN05` joins `MMA` and `WGMMA` in the enum. Three-way dispatch
at `humming.cuh:71` and `humming_ws.cuh:71` (just the non-WS path is wired
end-to-end; the WS path inherits the dispatch but is unexercised).

```
A: bf16 ────────────TMA/cp.async────►  smem.a            ──┐
                                                           │
B: uint4 codes ─────TMA/cp.async────►  smem.b   ──S2R──►  regs_qb ──dequant──► regs_b_tmp
                                                           │
                                                           │  (NEW for TCGEN05)
                                                           │  r2s with swizzle scatter
                                                           ▼
                                                       smem.b_dequant (bf16)
                                                           │
A from smem.a, B from smem.b_dequant ──tcgen05.mma──► TMEM accumulator
                                                           │
                                                           ▼
                                       t2r (tcgen05.ld) ──► RMEM ──► existing epilogue
```

Files we added or extended (everything else is unchanged):

| file                                            | role                                                        |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `humming/config/enum.py`                        | `MmaType.TCGEN05`                                          |
| `humming/config/mma.py`                         | `Tcgen05OpClassImpl` (codegen for the C++ MmaOpClass)      |
| `humming/config/config.py`                      | `TuningConfig.use_tcgen05`                                 |
| `humming/tune/sm100.py`                         | `Sm100Heuristics` (still inherits Sm89 -- no real tcgen05 tuning yet) |
| `humming/tune/__init__.py`                      | Register 100/101/102/103 in `heuristics_map`               |
| `humming/utils/device.py`                       | sm100 entries in `ops_map` for compute-bound threshold     |
| `humming/kernel/humming.py`                     | TCGEN05 MmaShape = (BlockM, BlockN, 16); warp_shape threaded into `from_config` |
| `humming/ops/utils.py`                          | CWD-resilient `humming.__file__` fallback                  |
| `include/humming/utils/enum.cuh`                | C++ `MmaType::TCGEN05`                                     |
| `include/humming/utils/storage.cuh`             | `b_dequant`, `tcgen05_tmem_col`, `tcgen05_mbar` under `IF_USE_TCGEN05` |
| `include/humming/utils/ptx/tcgen05.cuh`         | PTX wrappers (alloc/dealloc/mma/commit/ld/fence/instr_desc/smem_desc/elect_one_sync) |
| `include/humming/mma/tcgen05_mma.cuh`           | `TCGEN05` class (zero_accum / transform_b / run / final_regs_c_as_ptr) |
| `include/humming/kernel/humming.cuh`            | TMEM alloc + mbar init at entry; dealloc at exit           |
| `include/humming/kernel/humming_ws.cuh`         | Three-way dispatch (only)                                  |
| `include/humming/epilogue/{smem_reducer,smem_writer}.cuh` | `MAX(.., 1)` guards on WGMMA_CRegistersArrayType so it stays well-formed when MmaShape == BlockShape |
| `tests/test_sm100_smoke.py`                     | HummingLayer-level smoke through heuristic dispatch         |
| `tests/test_tcgen05.py`                         | Direct-construct TCGEN05 correctness test (xfail)          |
| `tests/bench_w4a16_baseline.py`                 | 3-way perf vs Sablefish + Marlin                           |
| `tests/baseline_phaseA.tsv`                     | Snapshot of humming's mma.sync perf, pre-tcgen05            |

## Current state

* `tests/test_sm100_smoke.py` (10 tests, mma.sync path): **passes**.
* `tests/baseline_phaseA.tsv`: humming's existing path is healthy across all
  shapes/batches, sets the floor we need to keep.
* `tests/test_tcgen05.py` (TCGEN05 path): **xfail**. Kernel compiles, runs
  to completion in ~4 s, output magnitudes match reference, but values are
  scrambled (~85% mismatched). Layout bug in the bf16 r2s.

## Sharp edges already paid for (don't re-discover)

Each of these cost real hours and would burn the next session if
forgotten. Citations are to the file in our checkout or the original spec.

* **SM100 SmemDescriptor layout != SM90's.** Bits and field meanings
  rearranged. Use the bit-by-bit builder in
  `include/humming/utils/ptx/tcgen05.cuh::tcgen05_smem_desc` (mirrors
  CUTLASS `cute/arch/mma_sm100_desc.hpp:98`). Humming's existing
  `make_wgmma_smem_desc` happens to produce a valid SM100 desc for
  128 B swizzle by coincidence -- don't rely on that.

* **`.sync.aligned` vs `elect_one_sync` is a hard split:**

  | instr                                           | issue style                |
  | ----------------------------------------------- | -------------------------- |
  | `tcgen05.alloc / dealloc / relinquish`          | all 32 warp lanes together |
  | `tcgen05.mma`                                   | `elect_one_sync` (one thread) |
  | `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64` | `elect_one_sync` |

  Mix them and the kernel hangs. The `.shared::cluster` qualifier on
  commit is mandatory (omitting it ptxas-accepts but the mbar never
  arrives). PTX 8.7 §9.7.16 is the spec; CUTLASS `cutlass/arch/barrier.h:770`
  is the canonical example.

* **`tcgen05.mma` requires the `{m0,m1,m2,m3}` mask operand.** Real form:
  `tcgen05.mma.cta_group::1.kind::f16 [d], a, b, idesc, {0,0,0,0}, p`.
  Without the mask, ptxas parses to a different variant; the hardware
  silently misinterprets and the MMA never retires (post-MMA mbar wait
  spins forever). CUTLASS `cute/arch/mma_sm100_umma.hpp:111`.

* **`tcgen05.mma.kind::f16` is K=16 per issue.** The mainloop must drive
  multiple issues with the SMEM pointer advancing by `iter_id *
  kKChunkUint128` (= 2 uint128_t = 16 bf16) to cover BlockK. Otherwise
  every iter re-multiplies the same first K-chunk.

* **TMEM accumulators are CTA-level.** Humming's existing mainloop
  assumes per-warp K-reduction via multiple K-warps; for TCGEN05 you
  must force `K_WARPS = 1` (i.e. `WarpShape::K == BlockShape::K`).
  Otherwise multiple K-warps stomp on the same TMEM column.

* **Instruction descriptor bit fields** (from CUTLASS
  `mma_sm100_desc.hpp:412`): `c_format` at bits [4,6), `a/b_format` at
  [7,13), `n_dim` at [17,23) (= N>>3), `m_dim` at [24,29) (= M>>4).
  The handcoded `union Tcgen05InstrDescriptor` in our `ptx/tcgen05.cuh`
  is the safe API; use it, don't reach for shift arithmetic.

* **MmaType.TCGEN05 codegen needs `warp_shape`.** Per-thread `CRegisters`
  count is `warp_M * warp_N * cd_bits / (32*32)`, not whole-block. Threaded
  through `MmaOpClass.from_config(warp_shape=...)` in `kernel/humming.py`.

## How to drive the test

```bash
# Smoke (heuristic dispatch path, mma.sync):
.venv/bin/python -m pytest humming/tests/test_sm100_smoke.py

# TCGEN05 path (xfail-marked until correctness lands):
rm -rf ~/.humming/cache    # forces fresh JIT after include edits
.venv/bin/python -m pytest humming/tests/test_tcgen05.py -s --runxfail

# Perf baseline (current humming + Sablefish + Marlin):
.venv/bin/python humming/tests/bench_w4a16_baseline.py
```

JIT recompile is ~10 s with empty cache, ~0 s with hits.

## Debug tools that work here

* **`compute-sanitizer --tool synccheck`** -- catches "Missing wait" on
  mbarrier mismatches, points at the failing SMEM address. Useful when
  the kernel hangs at a mbar.
* **`compute-sanitizer --tool memcheck`** -- illegal-memory access
  detection; less helpful when the hang has no actual memory error.
* **`nvidia-smi pmon -c 1 -s u`** during a hang -- if SM utilisation is
  99%, the kernel is in a busy loop (mbar_wait, most likely). If 0,
  the kernel never launched or is stuck on a synchronous host call.
* **Source-level bisection** -- comment out individual tcgen05 ops
  (`alloc`, `mma`, `commit`, `wait`, `t2r`, `dealloc`) and see which
  body causes the hang. Each compile-test cycle is ~10 s, faster than
  reasoning about which instruction is "supposed to" be wrong.
* **`cuobjdump --dump-sass ~/.humming/cache/<hash>/kernel.cubin`** to
  confirm `UTCATOMSWS` (alloc/dealloc), `UTCHMMA` (mma), `UTCBAR`
  (commit) are emitted in the expected count.
* **`nvcc --gpu-architecture=sm_103a -std=c++17 -ptx -I... kernel.cu`**
  to regenerate PTX from a cached kernel.cu for manual inspection.
  (cuobjdump can't disassemble SASS for instructions newer than its
  build; PTX always works.)

## Next step: r2s correctness

The kernel runs; only the layout of the bf16 staging in `smem.b_dequant`
is wrong. Output magnitudes match the reference exactly (mean ≈ max ≈
mean(|ref|)), only **which** (n, k) each value ends up at is broken.

Current attempt in `mma/tcgen05_mma.cuh::TCGEN05::run`: scatters each
thread's 32 bf16/K-iter to addresses computed from CUTLASS's m16n8k16
`BLayout = ((4,8),(2,2)):((16,1),(8,64))` (per
`mma_traits_sm80.hpp:89`), with `Swizzle<3,4,3>` XOR on the linear byte
offset. The math evidently doesn't match what tcgen05.mma reads through
its descriptor.

The concrete next step is **sentinel-write validation**:

1. Replace the r2s payload with `((threadIdx.x & 0xff) << 8) |
   (reg_index & 0xff)` (or similar) instead of the dequantised bf16.
2. Run a single MMA with **A = identity (just the bf16 representation
   of 1.0 along the diagonal)**, so each output `C[m,n] = sum_k A[m,k]
   * B[k,n] = B[m,n]`. The accumulator is then a direct view of how
   the hardware interpreted the B SMEM layout.
3. Read back via t2r and dump to GMEM. For each `out[m,n]` value,
   decode `(thread, reg)` and compare against expected.
4. The diff between expected and actual gives us the exact mapping.

Once the mapping is corrected, the kernel should produce correct bf16
output. **Don't try more blind XOR/stride permutations.** Each iteration
without instrumentation has not converged so far.

After correctness lands, the remaining items (none of them blocking
correctness) are:

* **TMEM column count is hardcoded `tcgen05_alloc<128>`** at kernel
  entry. Constexpr-derive from `BlockN * cd_bits / 32`.
* **`K_WARPS = 1` constraint.** OK for prototype; eventually want a
  tcgen05-aware mainloop that doesn't rely on humming's per-warp
  K-reduction.
* **Sm100Heuristics inherits Sm89.** Returns mma.sync configs. Once
  TCGEN05 is correct, write real heuristics: TCGEN05 above a per-shape
  batch crossover, MMA below.

## Quick file nav

```
/home/mgoin/code/vllm/humming/
├── workbook.md                                                   ← you are here
├── humming/include/humming/kernel/humming.cuh                    ← mainloop driver
├── humming/include/humming/mma/{wmma,wgmma,tcgen05_mma}.cuh      ← MMA classes
├── humming/include/humming/utils/ptx/tcgen05.cuh                 ← PTX wrappers
├── humming/include/humming/utils/storage.cuh                     ← SharedStorage
├── humming/config/{enum,config,mma}.py                           ← codegen
├── humming/kernel/humming.py                                     ← MMA op-class selection
├── humming/tune/sm100.py                                         ← heuristics
└── tests/{test_sm100_smoke,test_tcgen05,bench_w4a16_baseline}.py

/home/mgoin/code/vllm/.scratch/sablefish/                         ← perf reference
├── final_bench.py                                                ← target numbers
└── README.md                                                     ← CUTLASS baseline plumbing

/home/mgoin/code/vllm/.deps/cutlass-src/include/cute/             ← spec reference
├── arch/mma_sm100_desc.hpp                                       ← SmemDescriptor + InstrDescriptor
├── arch/mma_sm100_umma.hpp                                       ← canonical SM100_MMA_*_SS callers
├── arch/copy_sm100.hpp                                           ← TMEM_LOAD variants
└── atom/mma_traits_sm80.hpp:78                                   ← m16n8k16 BLayout
```
