#!/bin/bash
# r2 on a fresh pod, start to end (OUR script). Runs on the pod; logs to /workspace/runs/r2/prep.log.
#   1. Python env   2. unpack old + new clips   3. VAE-encode only the 10,815 new clips
#   4. re-encode all captions (new clips add new rows)   5. check the cache loads   6. train r2 like r1
set -euo pipefail
export HF_HOME=/workspace/hf
mkdir -p /workspace/runs/r2
step() { echo "=== $(date -u +%H:%M:%S) $*"; }

step "1 env"
python3 -m venv --system-site-packages /root/venv
/root/venv/bin/pip install -q safetensors diffusers av scikit-image transformers sentencepiece 2>&1 | grep -viE "warning|notice" || true

step "2 unpack"
mkdir -p /root/data
tar -xf /workspace/bhaav-clips.tar -C /root/data 2>/dev/null
tar -xf /workspace/bhaav-fill-r2.tar -C /root/data 2>/dev/null
cp -r /root/data/r2/splits /root/data/r2splits
find /root/data/clips -name "*.mp4" | wc -l

step "3 encode new latents"
cd /workspace
/root/venv/bin/python encode_latents.py --clips /root/data/clips --ids /root/data/r2splits/new_ids.txt \
    --out /root/new_latents --batch 16 --workers 16 2>&1 | grep -E "^shard|Error|Traceback" || true
mkdir -p /root/cache_r2
cp -r /workspace/cache/latents /root/cache_r2/latents
for f in /root/new_latents/shard_*.safetensors; do
    n=$(basename "$f" .safetensors | sed 's/shard_//')
    cp "$f" "/root/cache_r2/latents/shard_1${n:1}.safetensors"   # shard_00002 -> shard_10002: no clash with r1's shards
done
ls /root/cache_r2/latents | wc -l

step "4 encode captions (all clips)"
/root/venv/bin/python encode_captions.py --clips /root/data/clips --ids /root/data/r2splits/all_ids.txt \
    --out /root/cache_r2/text 2>&1 | grep -vE "[Ww]arn|deprecat" | tail -3
mkdir -p /workspace/cache_r2 && cp -r /root/cache_r2/text /root/new_latents /workspace/cache_r2/   # keep a copy on the volume

step "5 check the cache"
cd /workspace/model
/root/venv/bin/python -c "
from data import Cache
for s in ('train', 'val', 'test_seen', 'test_unseen'):
    c = Cache('/root/cache_r2', f'/root/data/r2splits/{s}.txt')
    print(s, len(c))
"

step "6 train r2"
/root/venv/bin/python train.py --cache /root/cache_r2 --splits /root/data/r2splits --out /workspace/runs/r2 --size s210 \
    --steps 40000 --warmup 1000 --decay_frac 0.25 --log_every 100 --val_every 1000 --ckpt_every 2500 --snap_every 5000 \
    --compile > /workspace/runs/r2/train.out 2>&1
step "done"
