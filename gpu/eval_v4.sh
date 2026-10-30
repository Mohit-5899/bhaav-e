#!/bin/bash
# Re-score r1 and r2 with effect checker v4 (OUR script). Runs on a fresh pod on the bhaav volume.
# Both runs regenerate the same 256 + 256 test stickers (same captions and seeds as eval v3) and are judged by the same
# checker files: r1's CLIP pool (eval_pool_3000.pt, already copied into runs/r2) and one shared effect v4 pool.
set -euo pipefail
export HF_HOME=/workspace/hf
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "1 env"
python3 -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -q safetensors diffusers av scikit-image scipy transformers sentencepiece 2>&1 | grep -viE "warning|notice" || true

step "2 r2 cache from the volume (symlinks: r1 latents + the 10,815 new ones, r2 captions)"
mkdir -p /root/cache_r2/latents
ln -sf /workspace/cache/latents/* /root/cache_r2/latents/
for f in /workspace/cache_r2/new_latents/shard_*.safetensors; do
    n=$(basename "$f" .safetensors | sed 's/shard_//')
    ln -sf "$f" "/root/cache_r2/latents/shard_1${n:1}.safetensors"
done
ln -sfn /workspace/cache_r2/text /root/cache_r2/text
ls /root/cache_r2/latents | wc -l

cd /workspace/model
step "3 r1 (builds the shared effect v4 pool from r1's training split)"
/root/venv/bin/python evaluate.py --run /workspace/runs/r1 --cache /workspace/cache --splits /workspace/splits \
    --meta /workspace/meta.jsonl --out /workspace/runs/r1/eval_v4 --n_gen 256 --n_pool 3000 --skip_loss \
    --guidance cfg:3,cfg:5 --effect_v4_pool /workspace/effect_v4_pool.pt --save_videos
step "4 r2"
/root/venv/bin/python evaluate.py --run /workspace/runs/r2 --cache /root/cache_r2 --splits /workspace/r2splits \
    --meta /workspace/meta.jsonl --out /workspace/runs/r2/eval_v4 --n_gen 256 --n_pool 3000 --skip_loss \
    --guidance cfg:3,cfg:5 --effect_v4_pool /workspace/effect_v4_pool.pt --save_videos
step "done"
