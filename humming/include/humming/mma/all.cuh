#pragma once

#include <humming/mma/mxmma.cuh>
#include <humming/mma/tcgen05_mma.cuh>
#include <humming/mma/tcgen05_ts_mma.cuh>
#include <humming/mma/wgmma.cuh>
#include <humming/mma/wmma.cuh>


template <MmaType kMmaType, class Ctx, class ArithClass>
struct MmaSelector;

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MMA, Ctx, ArithClass> {
  using Type = WMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::WGMMA, Ctx, ArithClass> {
  using Type = WGMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::TCGEN05, Ctx, ArithClass> {
  // SS = B staged in SMEM; TS = B staged in TMEM (opt-in via use_tcgen05_ts).
  using Type = std::conditional_t<Ctx::TuningConfig::kUseTcgen05Ts,
                                  TCGEN05_TS<Ctx, ArithClass>,
                                  TCGEN05<Ctx, ArithClass>>;
};

template <class Ctx, class ArithClass>
struct MmaSelector<MmaType::MXMMA, Ctx, ArithClass> {
  using Type = MXMMA<Ctx, ArithClass>;
};

template <class Ctx, class ArithClass>
using Mma = typename MmaSelector<Ctx::kMmaType, Ctx, ArithClass>::Type;
