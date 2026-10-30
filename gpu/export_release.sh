#!/bin/bash
# Package the final weights for Hugging Face and render the example stickers (OUR script). Runs on the pod.
# Weights: EMA copy only, fp32 safetensors (no pickle). Each release folder also carries the two small files sample.py
# reads (caption encoder settings, VAE latent statistics), so `--cache <folder>` works on a fresh download.
set -euo pipefail
export HF_HOME=/workspace/hf
cd /workspace/model
for run in r1 r2; do
    cache=/workspace/cache; [ $run = r2 ] && cache=/workspace/cache_r2
    out=/workspace/release/$run
    mkdir -p $out/text $out/latents
    cp $cache/text/info.json $out/text/ && cp /workspace/cache/latents/vae_config.json $out/latents/
    /root/venv/bin/python - "$run" "$out" <<'PY'
import json, sys, torch
from safetensors.torch import save_file
run, out = sys.argv[1:]
s = torch.load(f"/workspace/runs/{run}/snapshots/ema_step040000.pt", map_location="cpu")
save_file({k: v.float().contiguous() for k, v in s["ema"].items()}, f"{out}/bhaav_e.safetensors")
n = sum(v.numel() for v in s["ema"].values())
json.dump(dict(model="Bhaav-E VideoDiT", run=run, size=s["size"], step=s["step"], parameters=n, weights="EMA",
               frames=17, fps=8, resolution=256, vae="Wan-AI/Wan2.2-TI2V-5B-Diffusers (subfolder vae)",
               text_encoder="google/flan-t5-base", sampler=dict(steps=50, cfg=3.0, shift=5.0)),
          open(f"{out}/config.json", "w"), indent=1)
print(run, s["size"], s["step"], n)
PY
done
P=()
while IFS= read -r line; do P+=(--prompt "$line"); done <<'PROMPTS'
a crying emoji bouncing, tears falling, light blue background
a red furious emoji pulsing, steam puffing from its head, dark navy background
a blue success checkmark drawing itself inside a circle, confetti bursting out, white background
a party popper emoji pulsing, sparkles twinkling around it, light grey background
a cool emoji with sunglasses floating gently, sparkles twinkling around it, peach background
a glowing star emoji wobbling side to side, sparkles twinkling around it, dark navy background
a purple loading spinner spinning round and round, pulsing, black background
a screaming in shock emoji wobbling side to side, peach background
PROMPTS
/root/venv/bin/python sample.py --ckpt /workspace/release/r2/bhaav_e.safetensors --cache /workspace/release/r2 \
    "${P[@]}" --out /workspace/release/examples --seed 0
ls -la /workspace/release/r1 /workspace/release/r2 /workspace/release/examples
