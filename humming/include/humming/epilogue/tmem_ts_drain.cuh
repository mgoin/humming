#pragma once

#include <humming/datatype/base_conversion.cuh>
#include <humming/utils/base.cuh>
#include <humming/utils/ptx/tcgen05.cuh>

// Drain one warp's 32 TMEM lanes x kBlockM cols of the TS accumulator
// (transposed: lane = weight row n, col = activation m) into gmem_writer's
// smem.reduce layout:
//   smem_row = (n / 64) * kBlockM + m
//   int4 col = ((n / 8) % 8) ^ ((smem_row + smem_reduce_base) % 8)
// Each 8-lane group does an 8x8 register transpose so lane 8g + c ends up
// holding whole int4s; with i = n % 8 (i2i1i0) and c = m % 8 (c2c1c0) the three
// shfl.bfly stages swap i0<->c0 (half), i1<->c1 and i2<->c2 (reg bits).
// `n` is (warp % 4) * 32 + lane; `bias_val` is added per-n in f32 pre-convert.
// kApplyRowScale folds a channelwise weight scale in (constant along K, so the
// drain is a valid apply site); ElementC picks the f32 -> 16-bit convert.
template <uint32_t kBlockM, class ElementC = BFloat16,
          bool kApplyRowScale = false>
CUDA_INLINE void tmem_ts_drain_transposed(uint32_t d_base,
                                          uint32_t n,
                                          int4 *reduce,
                                          uint32_t smem_reduce_base,
                                          float bias_val,
                                          float scale_val = 1.0f) {
  uint32_t lane = threadIdx.x % 32u;
  uint32_t c = lane % 8u;                       // m offset after transpose
  uint32_t octet = n / 8u;                      // int4 column pre-swizzle
  uint32_t section_row_base = (n / 64u) * kBlockM;

  PRAGMA_UNROLL
  for (uint32_t chunk = 0; chunk < kBlockM / 32u; chunk++) {
    uint32_t tmp[32];
    tcgen05_ld_32x32b_x32(d_base + chunk * 32u, tmp);
    tcgen05_wait_ld();

    // f32 -> 16-bit pairs: v[p] = ElementC x2 (m = m0+2p, m0+2p+1) of row n.
    using Scalar2 = typename F16Conversion<ElementC>::scalar_t2;
    uint32_t v[16];
    PRAGMA_UNROLL
    for (uint32_t p = 0; p < 16u; p++) {
      float a0 = *reinterpret_cast<float *>(&tmp[2u * p]);
      float a1 = *reinterpret_cast<float *>(&tmp[2u * p + 1u]);
      if constexpr (kApplyRowScale) {
        a0 *= scale_val;
        a1 *= scale_val;
      }
      float2 f2 = make_float2(a0 + bias_val, a1 + bias_val);
      Scalar2 b2 = F16Conversion<ElementC>::float22num2(f2);
      v[p] = *reinterpret_cast<uint32_t *>(&b2);
    }

    // Stage 1: i0 <-> c0. Even lane keeps {own.lo, partner.lo}; odd
    // lane keeps {partner.hi, own.hi}.
    {
      bool lo = (lane & 1u) == 0u;
      PRAGMA_UNROLL
      for (uint32_t p = 0; p < 16u; p++) {
        uint32_t y = __shfl_xor_sync(0xffffffffu, v[p], 1);
        v[p] = lo ? __byte_perm(v[p], y, 0x5410) : __byte_perm(v[p], y, 0x3276);
      }
    }
    // Stage 2: i1 <-> c1 (reg pairs p, p^1). The i1=0 lane sends its
    // c1=1 reg and receives the partner's c1=0 reg, and vice versa.
    {
      bool lo = (lane & 2u) == 0u;
      PRAGMA_UNROLL
      for (uint32_t q = 0; q < 8u; q++) {
        uint32_t pe = 2u * q, po = pe + 1u;
        uint32_t y = __shfl_xor_sync(0xffffffffu, lo ? v[po] : v[pe], 2);
        (lo ? v[po] : v[pe]) = y;
      }
    }
    // Stage 3: i2 <-> c2 (reg pairs p, p^2).
    {
      bool lo = (lane & 4u) == 0u;
      PRAGMA_UNROLL
      for (uint32_t h = 0; h < 4u; h++) {
        PRAGMA_UNROLL
        for (uint32_t c1 = 0; c1 < 2u; c1++) {
          uint32_t pe = 4u * h + c1, po = pe + 2u;
          uint32_t y = __shfl_xor_sync(0xffffffffu, lo ? v[po] : v[pe], 4);
          (lo ? v[po] : v[pe]) = y;
        }
      }
    }

    // Four 128-bit swizzled stores: h-th int4 = 8 n-values at
    // m = chunk*32 + 8h + c.
    PRAGMA_UNROLL
    for (uint32_t h = 0; h < 4u; h++) {
      uint32_t m = chunk * 32u + 8u * h + c;
      uint32_t smem_row = section_row_base + m;
      uint32_t col = (octet % 8u) ^ ((smem_row + smem_reduce_base) % 8u);
      reduce[smem_row * 8u + col] =
          *reinterpret_cast<int4 *>(&v[4u * h]);
    }
  }
}
