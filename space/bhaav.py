"""Bhaav-E text -> sticker pipeline for the demo Space.  (OUR code; same maths as model/sample.py.)

Loads once: the trained VideoDiT (bhaav_e.safetensors from the model repo), the frozen Flan-T5-Base encoder and the
frozen Wan2.2 VAE. Works on CPU or GPU. Per sticker:
  1. caption -> 35 x 768 vectors (T5), plus the empty caption once, for guidance
  2. start from noise [48, 5, 16, 16] and take N Euler steps along the straight path noise -> clean
     (both guidance passes share one batch of 2, which is faster on CPU than two calls)
  3. VAE-decode to 17 frames of 256 x 256 and write a looping GIF (frame 17 repeats frame 1, so 16 are kept)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from videodit import SIZES, VideoDiT

MODEL_REPO = "Mohit5899/bhaav-e"
VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def timesteps(n, shift):
    t = torch.linspace(1.0, 0.0, n + 1)
    return shift * t / (1 + (shift - 1) * t)


class Bhaav:
    def __init__(self, device=None):
        from diffusers import AutoencoderKLWan
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, T5EncoderModel

        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        get = lambda f: hf_hub_download(MODEL_REPO, f)
        cfg = json.loads(Path(get("config.json")).read_text())
        self.model = VideoDiT(SIZES[cfg["size"]]).to(self.dev).eval()
        self.model.load_state_dict({k: v.float() for k, v in load_file(get("bhaav_e.safetensors")).items()})

        info = json.loads(Path(get("text/info.json")).read_text())
        self.max_len = info["max_len"]
        self.tok = AutoTokenizer.from_pretrained(info["model"])
        # bf16 only where the GPU supports it (A100, L4, ...); a Colab T4 does not, so it runs in fp32 (fast enough)
        self.bf16 = self.dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8   # Ampere or newer
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        self.t5 = T5EncoderModel.from_pretrained(info["model"], torch_dtype=dtype).to(self.dev).eval()

        vae_cfg = json.loads(Path(get("latents/vae_config.json")).read_text())
        self.mean = torch.tensor(vae_cfg["latents_mean"], device=self.dev).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(vae_cfg["latents_std"], device=self.dev).view(1, -1, 1, 1, 1)
        self.vae = AutoencoderKLWan.from_pretrained(VAE_REPO, subfolder="vae", torch_dtype=torch.float32)
        self.vae = self.vae.to(self.dev).eval()
        self.empty = self.encode([""])

    @torch.no_grad()
    def encode(self, texts):
        t = self.tok(texts, padding="max_length", truncation=True, max_length=self.max_len,
                     return_tensors="pt").to(self.dev)
        h = self.t5(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state
        return (h * t.attention_mask[..., None]).float(), t.attention_mask

    @torch.no_grad()
    def generate(self, prompt, steps=50, cfg=3.0, shift=5.0, seed=0, on_step=None):
        """Returns 16 uint8 frames [16, 256, 256, 3]. on_step(i, n) is called after each step (for a progress bar)."""
        txt, mask = self.encode([prompt])
        etxt, emask = self.empty
        both_txt, both_mask = torch.cat([txt, etxt]), torch.cat([mask, emask])
        c = self.model.cfg
        x = torch.randn(1, c.in_ch, c.frames, c.size, c.size,
                        generator=torch.Generator().manual_seed(int(seed))).to(self.dev)
        given = torch.zeros(2, dtype=torch.bool, device=self.dev)
        ts = timesteps(steps, shift).to(self.dev)
        auto = torch.autocast("cuda", dtype=torch.bfloat16) if self.bf16 else torch.autocast("cpu", enabled=False)
        with auto:
            for i in range(steps):
                v = self.model(x.expand(2, -1, -1, -1, -1), ts[i].expand(2), both_txt, both_mask, given)
                v_c, v_u = v[:1], v[1:]
                x = x - (v_u + cfg * (v_c - v_u)) * (ts[i] - ts[i + 1])
                if on_step:
                    on_step(i + 1, steps)
        video = self.vae.decode(x.float() * self.std + self.mean).sample          # [1, 3, 17, 256, 256] in [-1, 1]
        frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()
        return frames[:-1]


def save_gif(frames, path):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=125, loop=0)
    return path


if __name__ == "__main__":                       # timing check: python bhaav.py [threads] [steps]
    import sys
    threads, steps = int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if threads:
        torch.set_num_threads(threads)
    t0 = time.time(); b = Bhaav(); t1 = time.time()
    last = [0.0]
    def tick(i, s):
        last[0] = time.time()
    fr = b.generate("a crying emoji bouncing, tears falling, light blue background", steps=steps, on_step=tick)
    t2 = time.time()
    save_gif(fr, f"timing_{steps}steps.gif")
    print(f"threads {torch.get_num_threads()}  load {t1 - t0:.1f}s  denoise {last[0] - t1:.1f}s for {steps} steps "
          f"({(last[0] - t1) / steps:.2f}s/step)  decode {t2 - last[0]:.1f}s  total {t2 - t1:.1f}s")
