# Nemotron-3-Super-120B-A12B NVFP4 on GB10 with SGLang Native MTP

This repository is a reproducibility pack for a confirmed working NVFP4 run of NVIDIA Nemotron-3-Super-120B-A12B-NVFP4 on NVIDIA GB10 / Blackwell SM121 using SGLang native built-in MTP.

Credit: @mr-r0b0t on X — r0b0tlab.

## Headline result

- Served model: `NVIDIA/Nemotron-3-Super-120B-A12B-NVFP4`
- Runtime image: SGLang Nemotron dev image `lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3`
- Quantization: NVIDIA ModelOpt NVFP4
- Backend: `flashinfer_cutlass`
- Reasoning parser: `nemotron_3`
- Non-MTP baseline: about 14.84 tok/s
- Native MTP: about 21.64 tok/s
- Speedup: about 45.8 percent
- MTP accept rates: about 0.93 to 0.99
- Stability: tuned run completed without new NVRM/Xid faults

## Critical launch guidance

Use the built-in MTP layer. Do not pass the 120B model as `--speculative-draft-model-path`; that double-loads weights and can recreate host memory pressure.

Use a conservative GB10 profile:

- `SGLANG_ENABLE_SPEC_V2=1`
- `--speculative-algorithm NEXTN`
- `--disable-radix-cache`
- `--mem-fraction-static 0.72`
- `--max-running-requests 1`
- `--context-length 8192`
- `--fp4-gemm-backend flashinfer_cutlass`
- `--moe-runner-backend flashinfer_cutlass`

## Reproduce

Download the model separately. Do not commit weights into this repo.

```bash
export MODEL_HOST=/path/to/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
./scripts/launch_super120_mtp.sh
```

Run the smoke client if included:

```bash
python3 scripts/smoke_openai.py
```

## Evidence

See `results/` for the MTP results, root-cause notes, smoke JSON, long decode JSON, and kernel/counter summaries. The evidence board HTML is in `evidence/`.

## Known issues

The main risk is host/GPU memory pressure during 120B load, JIT, and CUDA graph setup. Stop ComfyUI, vLLM, other SGLang servers, Chromium previews, and GPU-heavy jobs before launch. Avoid rapid restart loops after driver errors.

## License

Scripts and documentation: MIT. Model weights are not redistributed here; follow upstream model license.
