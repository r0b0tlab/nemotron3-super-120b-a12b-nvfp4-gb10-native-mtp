# Root cause — Nemotron-3-Super-120B-A12B-NVFP4 freezes on GB10

## Evidence collected

Evidence directory:

`<local-path>

Important kernel evidence:

- Previous boot shows repeated NVIDIA RM allocation failures:
  - `NV_ERR_NO_MEMORY` from `_memdescAllocInternal`
  - `kgrctxAllocMainCtxBuffer`
  - `kgrctxAllocCtxBuffers`
  - repeated `NVRM: VM: invalid mmap`
- The same period shows CUDA compiler processes being OOM-killed:
  - `cudafe++ invoked oom-killer`
  - `cicc invoked oom-killer`
- Earlier previous boot also shows GPU fault after memory pressure:
  - `Xid 13 ... Illegal Instruction Parameter`
  - `Xid 43 ... name=python3`

Model metadata:

- Model: `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- Local size: ~75 GiB
- Architecture: `NemotronHForCausalLM`
- Layers: 88
- Hidden size: 4096
- Routed experts: 512
- Experts per token: 22
- Mamba heads: 128
- Context advertised by config: 262144
- Quantization: ModelOpt mixed precision; ~40,961 NVFP4 expert targets plus 139 FP8 targets

## Root cause

The freezes were memory-pressure failures, not ordinary request-level model errors.

The system was trying to serve a 75 GiB hybrid MoE/Mamba NVFP4 model while also compiling CUDA kernels / CUDA graph pieces. The CUDA compiler children (`cicc`, `cudafe++`) consumed several GiB each and were OOM-killed. At the same time, the NVIDIA driver failed to allocate graphics/context buffers (`kgrctx*`, `_memdescAllocInternal`) and then emitted invalid mmap messages. One prior run progressed to GPU exceptions (`Xid 13`, then `Xid 43`).

That pattern explains the host freezes:

1. 120B model weights leave limited headroom on 121 GiB unified memory.
2. JIT compilation / graph capture creates many transient host+GPU allocations.
3. Extra processes such as ComfyUI or other servers reduce margin further.
4. Driver allocation failures cascade into invalid mmap / GPU channel errors.

## Implementation direction

Use SGLang, not vLLM, for Nemotron Super on this host for now. The verified working path is the Nemotron-specific SGLang image:

`lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3`

Safe serving profile:

- `--quantization modelopt_fp4`
- `--fp4-gemm-backend flashinfer_cutlass`
- `--moe-runner-backend flashinfer_cutlass`
- `--reasoning-parser nemotron_3`
- `--max-running-requests 1`
- `--context-length 8192`
- `--mem-fraction-static 0.80`
- `--disable-piecewise-cuda-graph`
- stop ComfyUI / vLLM / old SGLang before launch

The project launch script previously used stale/unsafe assumptions (`super_v3` parser in one file, lower mem fraction that may not leave room for weights, and no standardized interference cleanup). The new implementation scripts fix those points.
