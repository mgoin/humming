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
  auto mainloop_arith = MainloopArithmetic();
  auto epilogue_arith = EpilogueArithmetic();
  auto mma = MMA(ctx, mainloop_arith);
  auto epilogue = Epilogue(ctx, epilogue_arith);
  auto producer = ProducerPipeline(ctx);
  auto consumer = ConsumerPipeline(ctx);
  auto s2r_pipe = S2RMemoryPipeline(ctx, mma, epilogue);

  producer.init_mbarrier();
  __syncthreads();

  // TMEM allocation + mbarrier init for the tcgen05 path. The tcgen05.*
  // family is .sync.aligned -- must be executed warp-uniformly. Use
  // warp 0 (threadIdx.x < 32) so all 32 threads issue together. Thread 0
  // also runs the mbarrier init (a regular intrinsic, not warp-aligned).
  // Size for now: 128 columns (covers BlockN up to 128 with f32 acc).
  // `.sync.aligned` instructions (tcgen05.alloc/dealloc/relinquish)
  // require ALL 32 threads of the issuing warp to execute together,
  // unlike tcgen05.mma/commit which use the elect_one_sync pattern.
  // Confirmed by bisection: with elect_one_sync the alloc spins.
  if constexpr (Ctx::kMmaType == MmaType::TCGEN05) {
    if (threadIdx.x < 32) {
      uint32_t smem_addr =
          cast_smem_ptr_to_uint(&smem.tcgen05_tmem_col);
      tcgen05_alloc<128>(smem_addr);
    }
    if (threadIdx.x == 0) {
      __mbarrier_init(&smem.tcgen05_mbar, /*expected_count=*/1);
    }
    __syncthreads();
  }

  while (scheduler.get_next_block()) {
    mma.zero_accum();
    __syncthreads();

    uint32_t &slice_iters = scheduler.slice_iters;
    producer.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.k_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.seek(scheduler.expert_id, scheduler.m_block_id, scheduler.n_block_id, scheduler.current_shape_m, scheduler.m_offset);
    epilogue.set_streamk_state(scheduler.slice_count, scheduler.slice_id, scheduler.locks_offset);

    if constexpr (TuningConfig::kUseTmaC) tma_wait_store_group<0, true>();
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0);

    while (slice_iters) {
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        if (slice_iters == 1) producer.load_channel();

        // TCGEN05 with kNumStages == 2 cannot start the next producer.load_stage
        // for the CURRENT stage at warp_iter_id == kWarpIters - 2, because
        // the WMMA / WGMMA paths rely on s2r having already pulled the stage's
        // A operand into RMEM by that point -- TCGEN05.mma reads A from SMEM
        // directly via the SS descriptor, so an in-flight cp.async overwrite
        // of smem.stages[stage_id].a races with the descriptor read of the
        // last K-iter. Defer the load (and its paired sync) by one iter for
        // that case so SMEM A is stable through the final mma.run; verified
        // empirically by the single-K-position probe showing only k=48..63
        // / k=112..127 / k=240..255 (the last 16 K of each K-block) returning
        // wrong output. The wait_stage stays at iter `kWarpIters - 2` because
        // it also gates the s2r reads of the NEXT stage at iter
        // `kWarpIters - 1`.
        constexpr bool kIsTcgen05TwoStage =
            Ctx::kMmaType == MmaType::TCGEN05 && kNumStages == 2;

        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
              if constexpr (!kIsTcgen05TwoStage) {
                producer.load_stage(stage_id, slice_iters > kNumStages);
              }
            } else {
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }
          if constexpr (kIsTcgen05TwoStage) {
            if (warp_iter_id == Ctx::kWarpIters - 1) {
              __syncthreads();
              producer.load_stage(stage_id, slice_iters > kNumStages);
            }
          }

          mma.transform_b((warp_iter_id + 1) % 2);
        }

        slice_iters--;
        if (!slice_iters) break;
      };
    };

    consumer.wait_channel();
    s2r_pipe.load_channel(scheduler.slice_id);
    __syncthreads();
    epilogue.call(mma.final_regs_c_as_ptr());
  }

  // Release the TMEM column allocation before kernel exit. tcgen05.*
  // is .sync.aligned -- must be warp-uniform.
  // dealloc + relinquish are also .sync.aligned -> warp-uniform.
  if constexpr (Ctx::kMmaType == MmaType::TCGEN05) {
    __syncthreads();
    if (threadIdx.x < 32) {
      tcgen05_relinquish_alloc_permit();
      tcgen05_dealloc<128>(smem.tcgen05_tmem_col);
    }
  }
};
