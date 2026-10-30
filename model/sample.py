"""Make stickers with a trained Bhaav Tiny checkpoint.  (OUR code.)

Generation = walk back along the straight line from noise (t = 1) to clean (t = 0):
    repeat 50 times:  x <- x - v * (t_now - t_next)
Two standard extras:
  - guidance (CFG): ask the model twice, with the caption and with the empty caption, then push further in the
    direction the caption adds:  v = v_empty + w * (v_caption - v_empty).  Bigger w = follows the words harder.
  - time shift: spend more of the 50 steps at high noise, where the overall layout is decided:
    t' = s * t / (1 + (s - 1) * t).

Usage (pod):
  python sample.py --ckpt /workspace/runs/r1/last.pt --cache /workspace/cache \
      --prompt "an angry emoji shaking, steam puffing from its head, black background" --out /workspace/samples
Usage (released weights from Hugging Face: bhaav_tiny.safetensors + config.json + text/info.json + latents/vae_config.json):
  python sample.py --ckpt <download>/bhaav_tiny.safetensors --cache <download> --prompt "..." --out samples
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from videodit import SIZES, VideoDiT


def timesteps(n, shift):
    t = torch.linspace(1.0, 0.0, n + 1)
    return shift * t / (1 + (shift - 1) * t)


@torch.no_grad()
def sample(model, txt, mask, empty_txt, empty_mask, steps=50, cfg=3.0, shift=5.0, first=None, seed=0,
           guide="cfg", bad_model=None, zero_steps=2):
    """txt/mask: [B, L, 768] caption vectors. first: optional clean latent frame [B, 48, 16, 16] (image -> sticker).
    guide picks how the two answers are combined (cfg = the guidance weight w in every mode):
      cfg      v = v_empty + w * (v_caption - v_empty)                          the standard recipe
      cfgzero  CFG-Zero* (Fan et al. 2025): first rescale v_empty by s = <v_caption, v_empty> / |v_empty|^2, the
               amount of v_empty that actually points along v_caption, then guide from there; and take no step at all
               for the first `zero_steps` steps, where an undertrained model's velocity is least reliable.
      auto     autoguidance (Karras et al. 2024): the "worse" answer comes from a weaker copy of the model (bad_model,
               e.g. an early snapshot) given the SAME caption, not from the empty caption. Pushing away from the weak
               copy's mistakes, instead of away from "no caption", sharpens quality without exaggerating the caption."""
    c = model.cfg
    B, dev = txt.shape[0], txt.device
    x = torch.randn(B, c.in_ch, c.frames, c.size, c.size, generator=torch.Generator().manual_seed(seed)).to(dev)
    given = torch.full((B,), first is not None, device=dev)
    ts = timesteps(steps, shift).to(dev)
    for i in range(steps):
        if first is not None:
            x[:, :, 0] = first
        t = ts[i].expand(B)
        v_c = model(x, t, txt, mask, given)
        if guide == "auto":
            v_u = bad_model(x, t, txt, mask, given)
        else:
            v_u = model(x, t, empty_txt.expand(B, -1, -1), empty_mask.expand(B, -1), given)
        if guide == "cfgzero":
            if i < zero_steps:
                continue
            vc, vu = v_c.float().flatten(1), v_u.float().flatten(1)
            s = ((vc * vu).sum(1) / (vu * vu).sum(1).clamp(min=1e-8)).view(B, 1, 1, 1, 1).to(v_u.dtype)
            v_u = v_u * s
        x = x - (v_u + cfg * (v_c - v_u)) * (ts[i] - ts[i + 1])
    if first is not None:
        x[:, :, 0] = first
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prompt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)   # r1 eval v3: 3 beat 5 and 8 on unseen combinations
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, T5EncoderModel

    if a.ckpt.endswith(".safetensors"):                                # released weights: EMA only, size in config.json
        from safetensors.torch import load_file
        size = json.loads((Path(a.ckpt).parent / "config.json").read_text())["size"]
        ema = load_file(a.ckpt, device=dev)
    else:
        s = torch.load(a.ckpt, map_location=dev)
        size, ema = s["size"], s["ema"]
    model = VideoDiT(SIZES[size]).to(dev).eval()
    model.load_state_dict({k: v.float() for k, v in ema.items()})     # the smooth EMA copy is the one we use

    info = json.loads((Path(a.cache) / "text" / "info.json").read_text())
    tok = AutoTokenizer.from_pretrained(info["model"])
    t5 = T5EncoderModel.from_pretrained(info["model"], torch_dtype=torch.bfloat16).to(dev).eval()
    def encode(texts):
        t = tok(texts, padding="max_length", truncation=True, max_length=info["max_len"], return_tensors="pt").to(dev)
        return (t5(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state
                * t.attention_mask[..., None]).float(), t.attention_mask
    txt, mask = encode(a.prompt)
    etxt, emask = encode([""])

    vae_cfg = json.loads((Path(a.cache) / "latents" / "vae_config.json").read_text())
    mean = torch.tensor(vae_cfg["latents_mean"], device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae_cfg["latents_std"], device=dev).view(1, -1, 1, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = sample(model, txt, mask, etxt, emask, a.steps, a.cfg, a.shift, seed=a.seed)
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae",
                                           torch_dtype=torch.float32).to(dev).eval()
    with torch.no_grad():                                                   # one clip at a time: 8 at once overflows a 24 GB GPU
        video = torch.cat([vae.decode(z[None]).sample for z in (x.float() * std + mean)])  # [B, 3, 17, 256, 256] in [-1, 1]
    frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu().numpy()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k, (p, clip) in enumerate(zip(a.prompt, frames)):
        imgs = [Image.fromarray(f) for f in clip[:-1]]                        # 16 frames: frame 17 repeats frame 1
        path = out / f"{k:02d}_seed{a.seed}.gif"
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=125, loop=0)
        print(path, "|", p)


if __name__ == "__main__":
    main()
