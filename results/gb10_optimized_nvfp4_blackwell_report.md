# GB10 optimized NVFP4 report

## Bottom line

NVFP4 is working on the GB10, and it is not just loading the checkpoint in some vague "quantized" mode. The active Nemotron-3-Super-120B server is using SGLang with ModelOpt NVFP4 weights, FlashInfer attention, and the FlashInfer/CUTLASS FP4 path selected for the heavy math. The MTP fast path is also live.

The current server is still running at:

`http://127.0.0.1:30000/v1`

Served model:

`nvidia/nemotron-3-super`

Active run:

`<local-nemotron-project>/runs/20260519T014530Z-super120-mtp-nextn-builtin-nvfp4`

Evidence bundle:

`<local-nemotron-project>/reports/blackwell_nvfp4_confirmation_20260519T021839Z.txt`

## What I confirmed

The live server reports healthy on `/health`, and `/v1/models` returns the expected served model with an 8192 token context window.

SGLang loaded the checkpoint through the ModelOpt path and explicitly detected NVFP4:

- `Using ModelOptModelLoader due to ModelOpt quantization config`
- `Detected nvfp4 checkpoint`
- runtime quantization: `modelopt_mixed`
- quant algo: `MIXED_PRECISION`

The selected runtime backends are the ones we want on Blackwell:

- attention backend: `flashinfer`
- FP4 GEMM backend: `flashinfer_cutlass`
- MoE backend: `flashinfer_cutlass`
- speculative MoE backend: `flashinfer_cutlass`

The active container also has generated FlashInfer/CUTLASS Blackwell artifacts in the cache. The important ones are under:

`/root/.cache/flashinfer/0.6.8.post1/121a/generated/`

The cache includes:

- `gen_gemm_sm120_cutlass_fp4`
- `fp4_gemm_cutlass___nv_bfloat16_128_128_128.cu`
- `fp4_gemm_cutlass___nv_bfloat16_128_128_256.cu`
- `fp4_gemm_cutlass___nv_bfloat16_256_128_128.cu`
- `cutlass_kernel_file_gemm_grouped_sm120_M256_BS_group0.generated.cu`
- `cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group*.generated.cu`

That is the part I care about most. We are not just seeing a config flag. The runtime has generated SM120 FP4 CUTLASS kernels in the active container cache, and SGLang is configured to use the FlashInfer/CUTLASS FP4 path.

Update after root profiling: Nsight Compute counter access works under root/privileged execution. A targeted FlashInfer `mm_fp4(..., backend="cutlass", use_nvfp4=True)` probe in the same SGLang/Nemotron image produced an SM120 block-scaled FP4 CUTLASS kernel with `SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, 16>`. NCU counted **524,288 tensor-pipe / HMMA-family instructions** and **8,589,934,592 FP4→FP32 tensor-path ops** per profiled GEMM iteration on GB10 CC 12.1. That closes the hardware-counter proof for the Blackwell FP4 tensor path. I am still not claiming a full roofline for the entire live 120B decode loop, because that would require relaunching the 120B server under `ncu`.

## Super 120B NVFP4 is stable now

The earlier freezes were not mysterious model failures. They were memory pressure cascades during 120B load, JIT, graph capture, or accidental double-loading. The signature was ugly:

- NVIDIA RM `NV_ERR_NO_MEMORY`
- `kgrctx*` context allocation failures
- `NVRM: VM: invalid mmap`
- OOM-killed CUDA compiler processes like `cicc` and `cudafe++`
- earlier `Xid 13` and `Xid 43` after pressure built up

The working path avoids that. ComfyUI, vLLM, stale SGLang containers, and profilers are stopped first. The MTP launcher also avoids the big trap: it does not pass the 120B model as `--speculative-draft-model-path`. Doing that starts loading a second copy of the 120B checkpoint and walks straight back into memory pressure. The working MTP path lets SGLang load the built-in `NemotronHForCausalLMMTP` layer instead.

## Current working Super 120B MTP profile

The current fast path is:

```bash
cd <local-nemotron-project>
./scripts/launch_super120_mtp.sh
```

Key settings:

```bash
SGLANG_ENABLE_SPEC_V2=1
--quantization modelopt_fp4
--fp4-gemm-backend flashinfer_cutlass
--moe-runner-backend flashinfer_cutlass
--speculative-algorithm NEXTN
--speculative-num-steps 1
--speculative-eagle-topk 1
--speculative-num-draft-tokens 2
--speculative-draft-model-quantization modelopt_fp4
--speculative-moe-runner-backend flashinfer_cutlass
--disable-radix-cache
--mem-fraction-static 0.72
--max-running-requests 1
--context-length 8192
--reasoning-parser nemotron_3
--disable-piecewise-cuda-graph
```

Two quirks matter:

First, SGLang accepts `NEXTN`, then internally reports it as `EAGLE`. That is expected.

Second, radix cache has to be off for NemotronH MTP. The default path fails with the Mamba scheduler, and `extra_buffer` is not supported for this model. `--disable-radix-cache` is the working fix.

## Startup numbers

Main model:

- type: `NemotronHForCausalLM`
- load time: 418.33 seconds
- model memory: 75.74 GiB
- available memory after main load: 39.61 GiB
- KV cache: 1,775,159 tokens
- K cache: 3.39 GiB
- V cache: 3.39 GiB
- CUDA graph capture: 17.37 seconds
- CUDA graph memory: 2.16 GiB

Built-in MTP draft layer:

- type: `NemotronHForCausalLMMTP`
- load time: 89.16 seconds
- draft memory: 7.08 GiB
- draft KV cache: 1,775,159 tokens
- draft K cache: 0.42 GiB
- draft V cache: 0.42 GiB

That is a good trade. The draft layer costs about 7 GiB, not another 75 GiB.

## Correctness and runtime behavior

Smoke tests passed:

- arithmetic: OK, 42 completion tokens in 2.25 seconds
- exact string: OK, 57 completion tokens in 2.75 seconds

The long decode test also passed. It returned the correct compact Python Fibonacci function.

Long decode result:

- completion tokens: 160
- reasoning tokens: 125
- elapsed: 7.39 seconds
- throughput: 21.64 tok/s

The decode logs show MTP is actually being used:

- accept length 1.98, accept rate 0.99
- accept length 1.85, accept rate 0.93
- accept length 1.93, accept rate 0.96

A later decode log reported 23.00 token/s generation throughput.

## Gains that matter

The big Super 120B win is MTP on top of stable NVFP4 serving.

Non-MTP safe baseline:

- 14.84 tok/s on the same long decode style workload

MTP NVFP4 fast path:

- 21.64 tok/s

That is a 1.46x speedup, or about +45.8%.

This is the first Super 120B path I would call genuinely usable on this GB10. It keeps the full 120B model on the box, uses the NVFP4 checkpoint, avoids the freeze path, and gets a meaningful decode uplift from the built-in MTP layer.

There is also a separate Nemotron-3-Nano result worth keeping in the story because it proves the same native Blackwell direction scales down cleanly. On the 30B/A3B NVFP4 profile, moving from the conservative native profile to the u80 FlashInfer/CUTLASS profile gave:

- KV capacity increase from 4.79M tokens to 8.31M tokens
- c64 throughput from 975.45 tok/s to 1040.08 tok/s versus the halfway profile, about +6.6%
- peak aggregate throughput around 1136.58 tok/s at c128
- no Marlin kernels in the Nsight Systems summary from the profiled native run
- SM120 block-scaled FP4 CUTLASS kernels hot in the trace

That 30B Nsight trace matters because it gives kernel-level proof that the Blackwell FP4 path can be hot. The 120B run adds the bigger practical point: the full Super model can now run, stay stable, and get real MTP speedup.

## Host health

The current server is healthy. NVIDIA SMI shows the GB10 in P0 with the SGLang scheduler using the GPU. Recent kernel logs from the successful MTP window have no new:

- `NVRM`
- `Xid`
- `OOM`
- `invalid mmap`
- `NV_ERR`
- `cicc` or `cudafe` OOM kills

Current host memory while the MTP server is running is tight but acceptable:

- about 22 GiB available
- about 3.4 GiB swap used

I would not run ComfyUI beside this. I would also avoid profiling, image generation, or another model server while Super 120B is up. The working profile is safe because it owns the machine.

## What is proven, and what is not

Proven:

- The Super 120B checkpoint is being loaded as NVFP4/ModelOpt mixed precision.
- SGLang is using FlashInfer attention.
- FP4 GEMM and MoE are set to FlashInfer/CUTLASS.
- The active container has generated SM120 FP4 CUTLASS kernels in the FlashInfer cache.
- The built-in MTP layer loads and runs.
- MTP acceptance is high, roughly 0.93 to 0.99 in the observed decode windows.
- Long decode throughput improved by about 45.8% over the non-MTP baseline.
- The host stayed clean during the successful run: no new Xids, OOM kills, or NVIDIA RM allocation errors.

Not yet proven:

- Exact tensor core utilization percentage.
- Exact SM occupancy.
- Full roofline position for the Super 120B MTP run.

That last part needs Nsight Compute counter access. Right now `ncu` cannot read NVIDIA performance counters as this user. If we enable that permission, the next move is a short c1 decode capture against the active MTP server or a profiler-launched server run, then inspect the FP4 MMA/CUTLASS kernels directly.

## My read

This is a real GB10 NVFP4 win.

The first milestone was getting out of the freeze loop. That meant isolating the host, stopping ComfyUI and other GPU users, disabling the risky paths, and not accidentally double-loading the 120B checkpoint. The second milestone was making the Blackwell path explicit: FlashInfer attention, FlashInfer/CUTLASS FP4 GEMM, FlashInfer/CUTLASS MoE, generated SM120 FP4 kernels in cache. The third milestone was turning on MTP without blowing the memory budget.

That last part is the payoff. Super 120B went from 14.84 tok/s to 21.64 tok/s on the long decode check, with a high accept rate and no new kernel errors. On a single GB10, that is the difference between "it technically runs" and "this is worth using."

## Recommended next steps

1. Keep this launcher as the canonical Super 120B fast path:

```bash
<local-nemotron-project>/scripts/launch_super120_mtp.sh
```

2. Do not run ComfyUI while Super 120B is live.

3. Treat `--speculative-draft-model-path` as dangerous for this model unless it points to a real small draft model. Passing the same 120B path is a footgun.

4. If we want the next round of proof, enable NVIDIA performance counters and run Nsight Compute. The target question is not "is NVFP4 working?" anymore. It is "how close are the SM120 FP4 kernels to the hardware ceiling?"
