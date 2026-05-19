# MTP / NEXTN result — Nemotron-3-Super-120B-A12B-NVFP4 on GB10

Run directory:

`<local-nemotron-project>/runs/20260519T014530Z-super120-mtp-nextn-builtin-nvfp4`

## Status

MTP is running successfully.

The active server is on:

`http://127.0.0.1:30000/v1`

Served model name:

`nvidia/nemotron-3-super`

## Working profile

- Runtime image: SGLang Nemotron dev image `lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3`
- Main model: `NemotronHForCausalLM`
- MTP draft: built-in model MTP layer, loaded as `NemotronHForCausalLMMTP`
- Quantization: `modelopt_fp4` / runtime reports `modelopt_mixed`
- FP4 backend: `flashinfer_cutlass`
- MoE backend: `flashinfer_cutlass`
- Speculative algorithm requested: `NEXTN`
- SGLang maps `NEXTN` to `EAGLE`
- Spec v2: enabled via `SGLANG_ENABLE_SPEC_V2=1`
- Radix cache: disabled for MTP compatibility on NemotronH
- `mem_fraction_static`: `0.72`
- `max_running_requests`: `1`
- `context_length`: `8192`
- draft steps/topk/tokens: `1 / 1 / 2`

## Gotchas discovered

1. `--speculative-algorithm NEXTN` is accepted, but SGLang internally remaps it to `EAGLE`.
2. With radix cache enabled, NemotronH speculative decode fails:
   - `Speculative decoding for NemotronHForCausalLM is not compatible with radix cache when using --mamba-scheduler-strategy no_buffer.`
3. Trying `--mamba-scheduler-strategy extra_buffer` also fails:
   - `mamba extra_buffer is not supported for NemotronHForCausalLM model`
4. Correct workaround: use `--disable-radix-cache`.
5. Do not pass `--speculative-draft-model-path` as the same 120B model. That double-loads the checkpoint and hits NVIDIA RM memory pressure. Let SGLang discover/use the built-in MTP layer.

## Startup metrics

Target model:

- Weight load: `418.33s`
- Model memory: `75.74 GiB`
- Available memory after main load: `39.61 GiB`
- Mamba cache:
  - `max_mamba_cache_size: 1`
  - conv state: `0.00 GiB`
  - SSM state: `0.31 GiB`
  - intermediate SSM cache: `0.62 GiB`
- KV cache:
  - `1,775,159 tokens`
  - K: `3.39 GiB`
  - V: `3.39 GiB`
- CUDA graph:
  - bs `[1]`
  - capture: `17.37s`
  - graph memory: `2.16 GiB`

MTP layer:

- Draft load type: `NemotronHForCausalLMMTP`
- Draft weight load: `89.16s`
- Draft memory: `7.08 GiB`
- Draft KV cache:
  - `1,775,159 tokens`
  - K: `0.42 GiB`
  - V: `0.42 GiB`

## Verification

Server readiness:

- `Application startup complete`
- `Uvicorn running on http://0.0.0.0:30000`
- `The server is fired up and ready to roll!`
- `/health`: OK

Smoke tests:

- arithmetic: OK, 42 completion tokens, 2.25s
- exact string: OK, 57 completion tokens, 2.75s

Long decode:

- prompt: compact Python Fibonacci function
- completion tokens: `160`
- reasoning tokens: `125`
- elapsed: `7.39s`
- throughput: `21.64 tok/s`
- output: correct compact Python `fib(n)` function

## MTP runtime evidence

SGLang decode logs show speculative acceptance:

- accept len: `1.98`, accept rate: `0.99`
- accept len: `1.85`, accept rate: `0.93`
- accept len: `1.93`, accept rate: `0.96`

Later decode throughput log:

- `gen throughput (token/s): 23.00`

Compared to previous non-MTP safe long decode:

- non-MTP: `14.84 tok/s`
- MTP: `21.64 tok/s`
- improvement: about `+45.8%`

## Kernel/host health

No `ERROR`, `Traceback`, `Killed`, `OOM`, or `OutOfMemory` lines in the run log after successful startup.

No new kernel danger lines found since launch for:

- `NVRM`
- `Xid`
- `OOM`
- `invalid mmap`
- `NV_ERR`
- `cicc`
- `cudafe`

Current resource profile while running:

- host memory: about `22 GiB` available
- swap: about `3.4 GiB` used
- container: about `3.5 GiB` host RSS reported by Docker, with unified GPU memory held through CUDA

## Active command

```bash
cd <local-nemotron-project>
./scripts/launch_super120_mtp.sh
```

The launcher uses the safe MTP settings above and first stops interfering GPU/model services.
