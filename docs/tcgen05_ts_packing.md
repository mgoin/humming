# tcgen05 TS-mode weight / scale / zero-point packing — specification

This is the packing spec for the TS-mode (TMEM-A) tcgen05 mainloop. §7 is the
interface contract for anyone writing a production packer against it; everything
else is the derivation and the executable-reference map.

Reference implementations (all validated against each other, see §8):

| artifact | where |
| --- | --- |
| Python/torch packer + inverse (weights, scales, zp) | `humming/utils/ts_packing.py` |
| CUDA repack variant | `weight_repack_nk<..., kUseTcgen05Ts=true>` in `humming/include/humming/kernel/process.cuh`, exposed as `ops.repack_weight(..., use_tcgen05_ts=True)` and selected for a layer by `mma_type="tcgen05"` |
| loader+dequant simulator | `simulate_ts_thread_regs` in `humming/utils/ts_packing.py` |
| tests | `tests/kernels/humming/test_tcgen05_packing.py` |

Scope: 16-bit activations (`kNumBitsA == 16`), power-of-2 even-bit B codes (u4
primary, u2/u8 covered). AWQ-style W4A16 (u4 + bf16 group scales + u4
zero-points, gs ∈ {64, 128}) is the production target. Odd-bit dtypes: see
decision points (§6).

---

## 1. Why the layout changes at all

Today's packer arranges codes so that after the lop3 dequant each thread's
regs form the **mma.sync m16n8k16 B fragment**: thread `t` owns weight rows
`n = t/4 + 8·f` and K positions `k = 2(t%4) + {0,1,8,9}` — ownership mixes
n and k across the warp.

TS-mode tcgen05 consumes dequantized weights from TMEM via `tcgen05.st`
(r2t), and TMEM A must be **K-major**: TMEM cell `(lane, col c)` =
`W[row, 2c .. 2c+1]`. A warp's `tcgen05.st.32x32b` writes lane-per-row, so
per-thread ownership must become **lane = row, full 16-K chunk per thread
per K-iter**. That is a different (n, k) → (thread, reg) map, hence a new
pack permutation. Everything else — file tensor shape, 16-K row tiling, g2s
path, TMA descriptors, `seek()` math — is deliberately kept identical to the
existing layout (§3).

## 2. The lop3 (i, i+4) interleave, re-derived for the TS destination

`uint_to_f16` (`humming/include/humming/datatype/dequant_single.cuh`)
dequants one 32-bit code word into `V/2` output regs, `V = 32 / kBitsB`
values per word (u4: V=8, u8: V=4). For reg `i` the word is shifted right by
`i·kBitsB` and masked with `(1 << kBitsB - 1) · 0x00010001`, so:

> reg `i` = ( value-slot `i` → lo bf16 , value-slot `i + V/2` → hi bf16 )

where "value-slot s" = bits `[s·kBitsB, (s+1)·kBitsB)` of the word. This is
the "(i, i+4) in-word interleave" for u4. It is a property of the dequant
instruction sequence, not of any layout — so the pack must pre-compensate
for **whatever** logical order we want the regs to come out in.

Wanted TS reg order (contract): reg `r` = bf16 pair `(K=2r, K=2r+1)`,
ascending K. Solving: logical K-ascending element `e` of a word must be
stored at value-slot

```
s(e) = (e % 2) * (V/2) + e / 2            (u4: V/2 = 4;  u8: V/2 = 2)
```

i.e. a u4 word's 8 nibbles hold, low-to-high: `[K0 K2 K4 K6 K1 K3 K5 K7]`
(K relative to the word's 8-K base). Inversely, slot `s` holds element
`e(s) = (s % (V/2))·2 + s/(V/2)`.

This is the **same** compensation formula the existing mma.sync pack uses
(`get_interleaved_index` in `process.cuh` with `stride = 32/kNumBitsA = 2`);
what changes is only *which (n, k)* is the logical element sequence. That is
why the CUDA TS variant reuses `humming_pack_weight` untouched (including
its 3/5/6/7-bit re-compression stages) and only replaces the gather that
fills the logical-order `tmp[]`.

Validation of this model: `test_mma_sync_inverse_matches_cuda_repack`
re-implements the existing layout in Python from this formula and matches
CUDA `ops.repack_weight` bit-exactly.

## 3. Packed weight tensor — file format

Identical outer shape/tiling to today's tensor:

```
int32 [ K/16 , N · 16 · kBitsB / 32 ]        (u4: [K/16, 2N])
```

* padded N: multiple of 64; padded K: multiple of 32 (same as today).
* row `c` holds K-chunk `[16c, 16c+16)` for ALL n — g2s slices columns
  `[n_block · BlockN·16·kBitsB/32, …)` per CTA tile exactly as today
  (`g2s_loader/loader_b.cuh` unchanged, TMA boxes unchanged).
* MoE: leading expert dim, unchanged.

Within one K-chunk row, define `W_r = 16·kBitsB/32` words per row per chunk
(u4: 2, u8: 4). **Forward map — code `W[n, k]` lives at:**

```
c   = k / 16                     packed row
j   = (k % 16) / V               word within the thread's W_r words
e   = (k % 16) % V               K-ascending element within the word
s   = (e % 2)·(V/2) + e/2        value slot (the §2 compensation)

B   = n / 64                     64-row block
l   = n % 32                     lane
h   = (n % 64) / 32              band half within the block

col = B·64·W_r + l·2·W_r + h·W_r + j
bits [ s·kBitsB , (s+1)·kBitsB )  of  packed[c, col]
```

Equivalently: each 64-row block is 32 **slots** of `2·W_r` words; slot `l`
(one 16-B slot for u4) = `[ row l : W_r words | row l+32 : W_r words ]`.

### Lane/row ownership tables (u4, one 64-row block, one K-chunk row)

Slot → words (u4, W_r = 2; word index within the block's 128 words):

| lane l | words 4l+0..1 | words 4l+2..3 |
| --- | --- | --- |
| 0 | row 0, K[0..8), K[8..16) | row 32, K[0..8), K[8..16) |
| 1 | row 1 | row 33 |
| … | … | … |
| 31 | row 31 | row 63 |

Nibble slots within a word (u4), K relative to the word's 8-K base:

| bits | [0,4) | [4,8) | [8,12) | [12,16) | [16,20) | [20,24) | [24,28) | [28,32) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K | 0 | 2 | 4 | 6 | 1 | 3 | 5 | 7 |

### Contrast with the mma.sync layout (for review sanity)

Existing layout (u4, decoded and pinned by tests):
`col = 128·(n/64) + 4·tid + (n%64)/16` with `tid = 4·(n%8) + (k%8)/2`,
slot from `e' = 4·((n%16)/8) + 2·((k%16)/8) + (k%2)`. Note `col` depends on
**both** n and k (fragment ownership); in the TS layout `col` depends only
on `(n, j)` — thread-owns-row is visible directly in the file format.

## 4. How the existing kernel loader delivers this (no loader change)

`s2r_loader/loader_b.cuh` at `WarpShape::N == 32` with 16-bit A activates
the **half-group path** (`kIsWarpHalfGroup`, and `kLoadHalfGroup` for
even-bit B): `kNumIntsPerThread = kBitsB/2` words per thread per K-iter
(u4: 2 words = one 8-B `uint2` load):

```
slot       = 32 · (n_warp_id / 2) + lane
LoadType#  = slot · 2 + (warp_id % 2)          (uint2 units, u4)
           = words [ slot·2·W_r + (warp_id%2)·W_r , +W_r )
```

With humming's warp decomposition (`context.cuh`: `n_warp_id = warp_id %
N_WARPS`, N fastest) and N_WARPS = BlockN/32 even, `warp_id % 2 ==
n_warp_id % 2` for every m/k-warp — so warp `w` (= n_warp covering band
`b = w % 4`), lane `l` receives exactly the words of row `32b + l` of its
64-row half-block: **the pack targets the unmodified loader byte-for-byte.**
`test_register_contract` walks every (warp, lane, K-chunk) through this
gather + the §2 slot extraction and asserts the contract.

`regs_qb` sizing note: the half-group path writes `kBitsB/2` words; the
existing `regs_qb[2][kBitsB]` declaration in the MMA structs is oversized
for TS but safe. The TS transform then runs `dequant(qb, res + 4j, j)` for
`j ∈ [0, kBitsB/2 / (V/2)…)` — for u4: j ∈ {0, 1}, producing regs 0..7 =
16 bf16 ascending K. No dequant code changes.

## 5. Scale / zero-point streams (lane = row ownership)

Today's streams give thread `t` the values for rows `t/4 + 8·f` (the
`transform_humming_weight_scale` 8×8 transpose within 64-row blocks +
`loader_bs`/`loader_bzp` `lane/4` indexing). With lane = row, each thread
needs its OWN row's scale/zp. New streams (natural order — the permutation
disappears entirely):

**Scales** (`pack_scales_tcgen05_ts`):

```
bf16 [ K/gs , N ]      — scale of (row n, group g) at packed[g, n]
```

Thread with row `n` reads one bf16 at index `n`. Adjacent lanes read
adjacent bf16 (2-B granule): lanes 2i/2i+1 share a 32-bit bank word →
conflict-free broadcast-merge. Optionally a future loader can vectorize by
having each lane load 2 groups (`uint32`) when BlockK spans 2 groups.

**Zero-points** (`pack_zero_point_tcgen05_ts`), zp_bits = 4 for kBitsB ≤ 4
else 8, V_z = 32/zp_bits:

```
int32 [ K/gs , N·zp_bits/32 ]
word w of group-row g = rows [w·V_z, (w+1)·V_z)
row n at bits [ (n % V_z)·zp_bits , +zp_bits )
```

Thread with row `n` reads word `n / V_z`, extracts slot `n % V_z`, and
broadcasts via `dequant_single_zero_point` (multiply by `0x00010001`) into
the per-call zp operand — all 4 zp_vals of a dequant call are identical in
TS mode (one row per thread), which *simplifies* `prepare_zp_for_dequant`
relative to today's fragment version.

**Bias** (epilogue): the TS drain reads the bias per weight row, so it stays
in natural `[N]` order rather than the C-fragment permutation.

## 6. Decision points

1. **BlockK > 64**: the B-code pack is 16-K-granular and BlockK-agnostic
   (the s2r loader steps `kSmemStride` per K-iter, sectionization never
   enters the packed format). The BlockK ≤ 64 constraint comes from the
   *activation* SMEM descriptor (`kSwizzleSizeK >= WarpK`), and the SS
   kernel already solves A-side sectionization. Recommendation: keep the
   pack as specified (no sectionization), solve BlockK > 64 on the
   descriptor side.
2. **loader_b half-group path — RESOLVED, slot-paired layout adopted.** The
   packed tensor above is byte-compatible with `loader_b`'s existing
   `WarpN == 32` half-group gather (§4), so the TS path needs no loader
   change and the file tensor keeps the shape, tiling and TMA descriptors of
   the shipped layout. The alternative — a band-major flat column formula —
   has a simpler forward map and a fully linear per-warp read, but requires
   a dedicated TS gather in `s2r_loader/loader_b.cuh`; it is a loader-side
   optimization that can be taken later without changing the contract's
   register semantics (§7), since only the column permutation differs.
3. **Odd-bit dtypes (u3/u5/u7)**: `kLoadHalfGroup` requires even kBitsB, so
   the half-group gather does not split odd-bit slots; and
   `humming_pack_weight`'s 3/5/6/7-bit re-compression crosses the row
   boundary inside a 32-element group under the TS element order (group =
   [row A 16K | row B 16K]). Both need a decision (dedicated loader path +
   in-group compression layout) before odd-bit TS support. u4/u8 (and by
   the same structure u2/u6/fp4/fp8-as-8bit) are covered.
4. **zp stream granularity**: nibble-packed words (compact, 1 shift+and per
   thread) vs byte-expanded (no extraction). Reference implements
   nibble-packed; switch is trivial if the transform warp turns out
   extraction-bound (it won't — it's 1 ALU op per K-group).
5. **Where scale application happens**: this spec only fixes *ownership*
   (lane = row). Whether scales are folded pre-`tcgen05.st` on the operand
   or in `transform_b` via `may_apply_bs_and_zp_on_b` (what the SS kernel
   does) is a mainloop decision and does not affect the streams.
6. **int2fp gating**: u4/bf16 never takes `should_preprocess_for_int2fp`
   (sign-magnitude) — the TS repack is value-preserving for the production
   path. The CUDA TS branch implements the int2fp/zp preprocessing anyway
   (same code, row-indexed), so u7/u8+zp style dtypes inherit it, but only
   u2/u4/u8 are test-pinned today.

## 7. Interface contract (paste-ready for the production packer)

> **tcgen05-TS register-layout contract (16-bit activations)**
>
> An MMA-M tile = `min(BlockN, 128)` weight rows (weight rows are
> humming's "N" dim). Within a 128-row tile, warp `w` of the
> `kWarpsPerMmaTile = MmaM/32` warps covering the tile owns rows
> `(w % 4) * 32 + lane` (lane = row within the warp's 32-row band).
>
> Per 16-K K-iter, after `dequant()` on the codes the pack produces,
> thread `lane` of warp `w` holds its single row's full 16-K bf16 chunk
> in 8 × uint32: **reg `r` = bf16 pair (K = 2r in lo half, K = 2r+1 in
> hi half), ascending K** — so TMEM cell `(lane, col c)` =
> `W[row, 2c..2c+1]`, the K-major layout TS-mode A requires. The lop3
> (i, i+4) in-word interleave is pre-compensated at pack time: value
> slot `s` of each packed word holds K-ascending element
> `(s % (V/2))*2 + s/(V/2)`, `V = 32/kBitsB`.
>
> Scales and zero-points follow lane = row ownership: scales
> `bf16 [K/gs, N]` natural order (thread reads index `row`);
> zero-points `int32 [K/gs, N*zp_bits/32]`, row `n` at bits
> `[(n % (32/zp_bits)) * zp_bits, ...)` of word `n / (32/zp_bits)`.
>
> File format: same tensor shape/tiling as the existing packed weight
> (`int32 [K/16, N*16*kBitsB/32]`); inside each 64-row block, 32 slots
> of `2*W_r` words: slot `l` = `[row l | row l+32]` half-slots of `W_r`
> words each, `W_r = 16*kBitsB/32` — byte-compatible with loader_b's
> WarpN==32 half-group gather, zero kernel loader changes.
>
> Everything downstream (tcgen05.st partition, TS-mode mma, epilogue) is
> on the kernel side. Executable reference: `humming/utils/ts_packing.py`
> (+ `weight_repack_nk<..., kUseTcgen05Ts=true>`), tests in
> `tests/kernels/humming/test_tcgen05_packing.py`.

## 8. Validation matrix

All in `tests/kernels/humming/test_tcgen05_packing.py`; the dtype-parametrized
cases run over {u2, u4, u8}:

| test | proves |
| --- | --- |
| `test_weight_pack_roundtrip` (6 shapes) + `_moe` | layout bijection |
| `test_register_contract` (every warp/lane/chunk) | loader gather + lop3 slots deliver the contract |
| `test_register_contract_block_n64` | MmaM=64 tiles (2-warp cover) |
| `test_scale_stream_roundtrip`, `test_zero_point_stream_roundtrip` | §5 streams |
| `test_mma_sync_inverse_matches_cuda_repack` (4 shapes) | the §2 slot model is bit-exact vs the shipped CUDA packer; both layouts encode identical weights |
| `test_cuda_repack_matches_reference` (7 shapes) | CUDA variant ≡ Python reference |
| `test_cuda_repack_packed_input`, `test_cuda_repack_moe` | end-to-end transform paths |
| `tests/kernels/humming/test_tcgen05.py` | the shared repack kernel change did not disturb the shipped SS path |
