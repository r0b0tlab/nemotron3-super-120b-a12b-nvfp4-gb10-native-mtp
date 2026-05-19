#!/usr/bin/env bash
set -euo pipefail

PROJECT=${ROOT:-$PWD}
MODEL_DIR=${MODEL_DIR:-${MODEL_HOST:-/path/to/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}}
PORT=${PORT:-30000}
# SGLang Nemotron dev image used to serve the Nemotron-3-Super checkpoint.
IMAGE=${IMAGE:-lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3}
RUN_DIR=${RUN_DIR:-$PROJECT/runs/$(date -u +%Y%m%dT%H%M%SZ)-super120-mtp-nvfp4}
CONTAINER=${CONTAINER:-sglang-nemotron-super-mtp}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-1}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
SPEC_STEPS=${SPEC_STEPS:-1}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-2}

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > /tmp/gb10_super120_mtp_run_dir.txt

"$PROJECT/scripts/stop_interference.sh" | tee "$RUN_DIR/preflight.log"

echo "[launch_mtp] run_dir=$RUN_DIR"
echo "[launch_mtp] model=$MODEL_DIR"
echo "[launch_mtp] image=$IMAGE"
echo "[launch_mtp] mem_fraction_static=$MEM_FRACTION_STATIC max_running_requests=$MAX_RUNNING_REQUESTS context_length=$CONTEXT_LENGTH algorithm=NEXTN steps=$SPEC_STEPS topk=$SPEC_TOPK draft_tokens=$SPEC_DRAFT_TOKENS"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

set +u
source <local-path> 2>/dev/null || true
set -u

exec docker run --rm --gpus all --ipc=host --network host \
  --name "$CONTAINER" \
  --shm-size 16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}" \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e MALLOC_ARENA_MAX=2 \
  -e OMP_NUM_THREADS=8 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -v "$MODEL_DIR:/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4:ro" \
  -v "$RUN_DIR:/evidence" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nvidia/nemotron-3-super \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --moe-runner-backend flashinfer_cutlass \
    --speculative-algorithm NEXTN \
    --speculative-num-steps "$SPEC_STEPS" \
    --speculative-eagle-topk "$SPEC_TOPK" \
    --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS" \
    --speculative-draft-model-quantization modelopt_fp4 \
    --speculative-moe-runner-backend flashinfer_cutlass \
    --disable-radix-cache \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --context-length "$CONTEXT_LENGTH" \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_3 \
    --disable-piecewise-cuda-graph \
    2>&1 | tee "$RUN_DIR/server.log"
