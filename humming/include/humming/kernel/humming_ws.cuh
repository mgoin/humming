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
    // TCGEN05 init (mirrors humming.cuh): the math side performs
    // `tcgen05.alloc<128>` from warp 0 and initialises the mbar for
    // tcgen05.commit. Both must complete before the first MMA issues;
    // the mbarrier_init_sync below publishes them (for cluster_size==1
    // it is a plain __syncthreads).
    if constexpr (Ctx::kMmaType == MmaType::TCGEN05) {
      if (threadIdx.x < 32) {
        uint32_t smem_addr =
            cast_smem_ptr_to_uint(&smem.tcgen05_tmem_col);
        tcgen05_alloc<128>(smem_addr);
      }
      if (threadIdx.x == 0) {
        __mbarrier_init(&smem.tcgen05_mbar, /*expected_count=*/1);
      }
    }
    mbarrier_init_sync<((TuningConfig::kMultiCastSizeA * TuningConfig::kMultiCastSizeB) > 1)>();
    consumer.arrive(kNumStages);

    while (scheduler.get_next_block()) {
      mma.zero_accum();

      uint32_t &slice_iters = scheduler.slice_iters;
      epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
      epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

      consumer.wait_stage<true>(kNumStages);
      s2r_pipe.load_stage_iter<true>(0, 0);
      mma.transform_b(0);

      while (slice_iters) {
        PRAGMA_UNROLL
        for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
          PRAGMA_UNROLL
          for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
            s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
            mma.run(stage_id, warp_iter_id);
            if (warp_iter_id == Ctx::kWarpIters - 2) {
#if !defined(TCGEN05_DEBUG_DEFER_ARRIVE)
              consumer.arrive(stage_id);
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
          }

          slice_iters--;
          if (!slice_iters) break;
        };
      };

      consumer.wait_channel();
      s2r_pipe.load_channel(scheduler.slice_id);

      if constexpr (kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);
      epilogue.call(mma.final_regs_c_as_ptr());
      // The TMA-store wait is PER-THREAD: only the issuing threads
      // (block_idx < BlockN/64 inside write_tma) track the bulk group.
      // Without the barrier the other math warps' lane-0s arrive on
      // math_mbar[kNumStages] while the TMA-C engine is still READING
      // smem.reduce -- which aliases stages[0] in the SMEM union -- so
      // the producer's next-block loads overwrite the bytes mid-read.
      if constexpr (TuningConfig::kUseTmaC) {
        tma_wait_store_group<0, true>();
        ctx.sync_math_threads();
      }
      if constexpr (!kReduceOverlapLastStageOnly) consumer.arrive(kNumStages);
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
        tcgen05_dealloc<128>(smem.tcgen05_tmem_col);
      }
    }
  }

  __syncthreads();
  if constexpr (TuningConfig::kMultiCastSizeA > 0 || TuningConfig::kMultiCastSizeB > 0) {
    asm volatile("barrier.cluster.arrive;\n");
    asm volatile("barrier.cluster.wait;\n");
  }
};
