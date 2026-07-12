#pragma once

#include <humming/scheduler.cuh>
#include <humming/utils/all.cuh>
#include <humming/utils/ptx/tcgen05.cuh>

#include <humming/arith/epilogue_arith.cuh>
#include <humming/arith/mainloop_arith.cuh>

#include <humming/epilogue/pipeline.cuh>
#include <humming/memory/g2s_pipeline.cuh>
#include <humming/memory/s2r_pipeline.cuh>
#include <humming/mma/all.cuh>

#include <humming/datatype/dequant.cuh>


template <bool kUseTma>
class KernelTensorParamType {
public:
  using Type = std::conditional_t<kUseTma, CUtensorMap const, void *const>;
};

CUDA_INLINE const void *param_to_ptr(const CUtensorMap &x) { return &x; }
CUDA_INLINE const void *param_to_ptr(void *const &x) { return x; }


template <
    class MmaOpClass,
    class ProblemShape, class BlockShape, class WarpShape, class PadShape,
    class ElementA, class ElementB, class ElementC, class ElementBS,
    class LayerConfig, class ComputeConfig, class TuningConfig>
__global__ __launch_bounds__(TuningConfig::kNumThreads, TuningConfig::kNumCtasPerSm) void humming(
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaA>::Type A,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaB>::Type B,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaC>::Type C,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaAS>::Type AS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBS>::Type BS,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBZP>::Type BZP,
    const __grid_constant__ typename KernelTensorParamType<TuningConfig::kUseTmaBias>::Type Bias,
    const uint32_t *GS,
    const uint32_t *sorted_ids_ptr,
    const uint32_t *expert_ids_ptr,
    const uint32_t *num_tokens_padded_ptr,
    const uint32_t *expert_layout_ptr,
    CUtensorMap *tensor_map_buffer,
    int32_t *locks,
    uint32_t shape_m,
    uint32_t top_k,
    bool use_int64_expert_layout) {

  constexpr uint32_t kNumThreads = TuningConfig::kNumThreads;
  constexpr uint32_t kNumStages = TuningConfig::kNumStages;
  constexpr bool kReduceOverlapLastStageOnly = TuningConfig::kReduceOverlapLastStageOnly;

  using SharedStorage = SharedStorage<
      MmaOpClass, BlockShape, WarpShape, ElementA, ElementB, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Ctx = KernelContext<
      MmaOpClass, ProblemShape, BlockShape, WarpShape, PadShape,
      ElementA, ElementB, ElementC, ElementBS,
      LayerConfig, ComputeConfig, TuningConfig>;
  using Scheduler = Scheduler<Ctx>;
  using ProducerPipeline = ProducerPipeline<Ctx>;
  using ConsumerPipeline = ConsumerPipeline<Ctx>;
  using MainloopArithmetic = MainloopArithmetic<Ctx>;
  using EpilogueArithmetic = EpilogueArithmetic<Ctx>;
  using MMA = Mma<Ctx, MainloopArithmetic>;
  using Epilogue = EpiloguePipeline<Ctx, MMA, EpilogueArithmetic>;
  using S2RMemoryPipeline = S2RMemoryPipeline<Ctx, MMA, Epilogue>;

  extern __shared__ int4 shared_memory[];
  auto &smem = *reinterpret_cast<SharedStorage *>(shared_memory);

  const KernelParams params{
      shape_m, top_k, use_int64_expert_layout,
      param_to_ptr(A), param_to_ptr(B), param_to_ptr(AS), param_to_ptr(BS),
      param_to_ptr(BZP), param_to_ptr(Bias), param_to_ptr(C), GS,
      sorted_ids_ptr, expert_ids_ptr, num_tokens_padded_ptr, expert_layout_ptr,
      tensor_map_buffer, locks};
  auto ctx = Ctx(smem, params);

  auto scheduler = Scheduler(ctx);
  if (ctx.is_load_thread()) {
    if constexpr (TuningConfig::kNumMathThreads > 256) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    } else if constexpr (TuningConfig::kNumCtasPerSm == 1 && ElementA::kBits != 16) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(40));
    } else {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(24));
    }

    auto producer = ProducerPipeline(ctx);
    producer.init_mbarrier();
    mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();
    while (scheduler.get_next_block()) {
      uint32_t &slice_iters = scheduler.slice_iters;

      producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
      producer.wait_math_epilogue();
      producer.load_stage<true, true>(0);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 1; stage_id < kNumStages - 1; stage_id++) {
        producer.load_stage(stage_id, stage_id < slice_iters);
      };

      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          if (slice_iters == 1) producer.load_channel();
          producer.wait_stage(stage_id);
          producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
          slice_iters--;
          if (!slice_iters) break;
        }
      }
    }
  } else {
    if constexpr (TuningConfig::kNumMathThreads > 256) {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(96));
    } else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(232));
    }

    auto mainloop_arith = MainloopArithmetic();
    auto epilogue_arith = EpilogueArithmetic();
    auto mma = MMA(ctx, mainloop_arith);
    auto epilogue = Epilogue(ctx, epilogue_arith);
    auto consumer = ConsumerPipeline(ctx);
    auto s2r_pipe = S2RMemoryPipeline(ctx, mma, epilogue);

    consumer.init_mbarrier();
    // TCGEN05 init (mirrors humming.cuh): the math side performs the
    // TMEM alloc from warp 0 and initialises the commit mbarrier(s).
    // Both must complete before the first MMA issues; the
    // mbarrier_init_sync below publishes them (for cluster_size==1 it
    // is a plain __syncthreads). With accumulator multi-staging the
    // alloc doubles to 256 cols (2 x BlockN <= 128 buffers).
    constexpr uint32_t kTcgen05AccStages = TuningConfig::kTcgen05AccStages;
    if constexpr (Ctx::kMmaType == MmaType::TCGEN05) {
      if (threadIdx.x < 32) {
        uint32_t smem_addr =
            cast_smem_ptr_to_uint(&smem.tcgen05_tmem_col);
        tcgen05_alloc<SharedStorage::kTcgen05TmemCols>(smem_addr);
      }
      if (threadIdx.x == 0) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kTcgen05AccStages; i++) {
          __mbarrier_init(&smem.tcgen05_mbar[i], /*expected_count=*/1);
        }
#if HUMMING_USE_TCGEN05_TS
        __mbarrier_init(&smem.tcgen05_ts_mbar[0], /*expected_count=*/1);
        __mbarrier_init(&smem.tcgen05_ts_mbar[1], /*expected_count=*/1);
#endif
      }
    }
    mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();
    consumer.arrive(kNumStages);

    // Deferred-epilogue state (TCGEN05 accumulator multi-staging only).
    // The pending tile's gmem coords + stream-k state are snapshotted at
    // commit time and replayed into `epilogue.seek` at drain time.
    uint32_t acc_buf = 0;
    bool pending = false;
    uint32_t pending_buf = 0;
    uint32_t p_expert = 0, p_m_blk = 0, p_n_blk = 0;
    uint32_t p_shape_m = 0, p_m_off = 0;
    uint32_t p_scount = 1, p_sid = 0, p_lockoff = 0;

    while (scheduler.get_next_block()) {
      if constexpr (kTcgen05AccStages > 1) mma.set_accum_buf(acc_buf);
      mma.zero_accum();

      uint32_t &slice_iters = scheduler.slice_iters;
      if constexpr (kTcgen05AccStages == 1) {
        epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
        epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);
      }

      consumer.wait_stage<true>(kNumStages);
      s2r_pipe.load_stage_iter<true>(0, 0);
      mma.transform_b(0);

      // TS mode + the arrive-at-kWarpIters-2 hazard (track A saw real
      // corruption from it at K=4096 on the SS ws-pipeline): the plain
      // mainloop is already safe WITHOUT deferring the release, by a
      // three-link chain that the static_asserts below pin:
      //   1. The producer overwrites stage T only after the math warps
      //      release stage T+1 (its loop refills stage
      //      (s + kNumStages - 1) % kNumStages upon wait_stage(s)).
      //   2. Each math warp's arrive of stage T+1 at (T+1, kWarpIters-2)
      //      is program-ordered after its transform_b(slot 1) at
      //      (T+1, iter 0) -- this needs kWarpIters >= 4.
      //   3. That transform_b's WAR wait awaited the COMPLETION of the
      //      tcgen05.commit issued at run(T, kWarpIters-1) -- a batch
      //      commit, so every MMA reading stage T provably retired.
      // Define TCGEN05_TS_DEFER_STAGE_RELEASE to instead release stage T
      // explicitly after transform_b(slot 1) of stage T+1 (track A's
      // deferral, needed if the geometry ever breaks link 2). Measured
      // cost on B300: ~3.3% at Llama70B-down M=2048 (2483 vs 2404 us) --
      // the delayed release stalls the TMA producer.
      constexpr bool kIsTcgen05Ts = TuningConfig::kUseTcgen05Ts;
      if constexpr (kIsTcgen05Ts) {
        static_assert(Ctx::kWarpIters >= 4 && Ctx::kWarpIters % 2 == 0,
                      "TS stage-release safety: transform_b(slot 1) of "
                      "stage T+1 must precede the stage-(T+1) arrive, and "
                      "slot 1 must own each stage's last commit");
      }
#if defined(TCGEN05_TS_DEFER_STAGE_RELEASE)
      constexpr bool kTsDeferredRelease = kIsTcgen05Ts;
      if constexpr (kTsDeferredRelease) {
        static_assert(kNumStages >= 3,
                      "TS deferred G2S release: releasing stage T during "
                      "stage T+1 deadlocks the producer handshake at 2 "
                      "stages");
      }
#else
      constexpr bool kTsDeferredRelease = false;
#endif
      uint32_t ts_prev_stage = 0;
      bool ts_has_prev = false;

      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          PRAGMA_UNROLL
          for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
            s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
            mma.run(stage_id, warp_iter_id);
            if (warp_iter_id == Ctx::kWarpIters - 2) {
#if !defined(TCGEN05_DEBUG_DEFER_ARRIVE)
              if constexpr (kTsDeferredRelease) {
                ts_prev_stage = stage_id;
                ts_has_prev = true;
              } else {
                consumer.arrive(stage_id);
              }
#else
              // TCGEN05 probe: the SS-mode MMAs read A from this
              // stage's SMEM asynchronously; releasing the stage here
              // lets the TMA producer overwrite A while the last two
              // MMAs may still read it. Defer arrive to after a full
              // MMA drain at the last warp-iter.
              if constexpr (Ctx::kMmaType != MmaType::TCGEN05)
                consumer.arrive(stage_id);
#endif
              if (slice_iters > 1) {
                consumer.wait_stage((stage_id + 1) % kNumStages);
              }
            }
#if defined(TCGEN05_DEBUG_DEFER_ARRIVE)
            if constexpr (Ctx::kMmaType == MmaType::TCGEN05) {
              if (warp_iter_id == Ctx::kWarpIters - 1) {
                mma.drain_mmas();
                consumer.arrive(stage_id);
              }
            }
#endif

            mma.transform_b((warp_iter_id + 1) % 2);
            if constexpr (kTsDeferredRelease) {
              if (warp_iter_id == 0 && ts_has_prev) {
                consumer.arrive(ts_prev_stage);
                ts_has_prev = false;
              }
            }
          }

          slice_iters--;
          if (!slice_iters) break;
        };
      };

      consumer.wait_channel();
      s2r_pipe.load_channel(scheduler.slice_id);

      if constexpr (kTcgen05AccStages > 1) {
        // Commit + wait this tile's MMA batch (the wait must precede
        // the producer release: in-flight tcgen05.mma reads stage SMEM
        // through the async proxy until it retires). Then release the
        // producer (safe: `smem.reduce` is a dedicated buffer, not
        // unioned with the stages) and drain the PREVIOUS tile's
        // accumulator -- the producer's TMA loads for the next tile
        // overlap the t2r + gmem write below.
        mma.commit_accum();
        mma.wait_accum();
// Bisection switch: define to keep shipped drain ordering while using
// all the new storage/alloc/rotation plumbing.
// #define TCGEN05_ACC2_NO_DEFER 1
#ifdef TCGEN05_ACC2_NO_DEFER
        // Bisection mode: shipped ordering (drain this tile now, arrive
        // after), keeping the new storage/alloc/rotation plumbing.
        epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
        epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);
        epilogue.call(mma.drain_accum());
        if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
        consumer.arrive(kNumStages);
        acc_buf ^= 1u;
#else
        consumer.arrive(kNumStages);

        if (pending) {
          mma.set_accum_buf(pending_buf);
          epilogue.seek(p_expert, p_m_blk, p_n_blk, p_shape_m, p_m_off);
          epilogue.set_streamk_state(p_scount, p_sid, p_lockoff);
          epilogue.call(mma.drain_accum());
          // The TMA-store wait is PER-THREAD (only the issuing thread
          // tracks the group); the barrier keeps the other math warps
          // from overwriting smem.reduce with the NEXT drain before
          // the store engine has read this one.
          if constexpr (TuningConfig::kUseTmaC) {
            tma_wait_store_group<0, true>();
            ctx.sync_math_threads();
          }
          pending = false;
        }
        if (scheduler.slice_count > 1) {
          // Stream-k tiles participate in cross-CTA lock chains;
          // deferring their drain can create lock-wait cycles between
          // CTAs whose pending drains block on each other. Drain
          // immediately instead (stream-k tiles are scheduled last, so
          // the deferral win is spent by then anyway).
          mma.set_accum_buf(acc_buf);
          epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
          epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);
          epilogue.call(mma.drain_accum());
          if constexpr (TuningConfig::kUseTmaC) {
            tma_wait_store_group<0, true>();
            ctx.sync_math_threads();
          }
        } else {
          pending = true;
          pending_buf = acc_buf;
          p_expert = scheduler.expert_id;
          p_m_blk = scheduler.m_block_id;
          p_n_blk = scheduler.n_block_id;
          p_shape_m = scheduler.current_shape_m;
          p_m_off = scheduler.m_offset;
          p_scount = scheduler.slice_count;
          p_sid = scheduler.slice_id;
          p_lockoff = scheduler.locks_offset;
          acc_buf ^= 1u;
        }
#endif
      } else {
        if constexpr (kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);
        epilogue.call(mma.final_regs_c_as_ptr());
        if constexpr (kTsDeferredRelease) {
          // The tile's final stage is still pending. Safe to release
          // now: final_regs_c_as_ptr waited the tcgen05 commit mbar,
          // so every MMA reading its SMEM has retired.
          if (ts_has_prev) consumer.arrive(ts_prev_stage);
        }
        if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
        if constexpr (!kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);
      }
    }
    if constexpr (kTcgen05AccStages > 1) {
      if (pending) {
        mma.set_accum_buf(pending_buf);
        epilogue.seek(p_expert, p_m_blk, p_n_blk, p_shape_m, p_m_off);
        epilogue.set_streamk_state(p_scount, p_sid, p_lockoff);
        epilogue.call(mma.drain_accum());
        if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
      }
    }
    // Release TMEM (mirrors humming.cuh). All 32 threads of warp 0 must
    // execute together since tcgen05.{dealloc, relinquish_alloc_permit}
    // are .sync.aligned. Sync only the math threads (barrier 1) so all
    // t2r reads retire before the dealloc. A plain __syncthreads here
    // would pair with the load threads' joint __syncthreads below and
    // shift the bar-0 pairing: the math threads' own joint sync would
    // then deadlock against load threads parked in the cluster barrier.
    if constexpr (Ctx::kMmaType == MmaType::TCGEN05) {
      ctx.sync_math_threads();
      if (threadIdx.x < 32) {
        tcgen05_relinquish_alloc_permit();
        tcgen05_dealloc<SharedStorage::kTcgen05TmemCols>(smem.tcgen05_tmem_col);
      }
    }
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA > 0 || TuningConfig::kMultiCastSizeB > 0) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};
