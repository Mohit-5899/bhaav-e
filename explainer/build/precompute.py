"""Step 1 of the explainer build: run Bhaav-E on the explainer's prompts and save every step.  (OUR code.)

The explainer page runs no model. Like Diffusion Explainer, it plays back results computed here in advance.
Each run = one prompt x one seed x one guidance scale. For every one of the 50 steps we save:
  xs[i]      the latent the sampler holds after step i (xs[0] is the starting noise)      [51, 48, 5, 16, 16]
  guess[i]   the model's guess of the finished clip at step i: x - t * v                  [50, 48, 5, 16, 16]
  vc[i], vu[i]  the velocity predicted with the caption and with the empty caption         [50, 48, 5, 16, 16]
and fully decode (frozen Wan2.2 VAE) the guess at a few checkpoint steps, because one decode takes ~14 s on an M1 Pro.
Same maths as space/bhaav.py; latents stay normalised (what the model sees).

Run (resumable; finished runs are skipped):
    HF_HUB_OFFLINE=1 ../../.venv/bin/python precompute.py --device mps
Writes raw/<run>.npz, raw/<run>_s<step>.jpg (16 frames side by side, 256 px) and raw/text.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "space"))
from bhaav import Bhaav, timesteps  # noqa: E402

# Pairs that differ in one attribute, so the page can compare them from the same starting noise.
PAIRS = [
    ("effect", "a crying emoji bouncing, tears falling, light blue background",
               "a crying emoji bouncing, light blue background"),
    ("motion", "a red furious emoji shaking, steam puffing from its head, dark navy background",
               "a red furious emoji pulsing, steam puffing from its head, dark navy background"),
    ("effect", "a party popper emoji bouncing, confetti bursting out, light grey background",
               "a party popper emoji bouncing, sparkles twinkling around it, light grey background"),
    ("background", "a cool emoji with sunglasses floating gently, sparkles twinkling around it, peach background",
                   "a cool emoji with sunglasses floating gently, sparkles twinkling around it, dark navy background"),
]
PROMPTS = [p for _, a, b in PAIRS for p in (a, b)]
SEEDS = [0, 1]
CFGS = [0.0, 1.0, 3.0, 6.0]
STEPS, SHIFT = 50, 5.0
DECODE_AT = [5, 10, 20, 30, 50]          # steps whose guess is fully decoded (one decode ~10 s on an M1 Pro)
FULL_CFG = 3.0                            # the default guidance scale, shown first


def grid():
    """All runs, in build order: the default view (seed 0, guidance 3) first, so the page can be built early."""
    keys = [(p, s, g) for s in SEEDS for g in CFGS for p in range(len(PROMPTS))]
    return sorted(keys, key=lambda k: (k[1], k[2] != FULL_CFG))


def run_id(p, seed, cfg):
    return f"p{p}_s{seed}_g{cfg:g}"


def to_strip(frames):
    """[16, H, W, 3] uint8 -> one image with the 16 frames side by side."""
    return Image.fromarray(np.concatenate(list(frames), axis=1))


@torch.no_grad()
def one_run(b, prompt, seed, cfg, out):
    txt, mask = b.encode([prompt])
    etxt, emask = b.empty
    both_txt, both_mask = torch.cat([txt, etxt]), torch.cat([mask, emask])
    c = b.model.cfg
    x = torch.randn(1, c.in_ch, c.frames, c.size, c.size,
                    generator=torch.Generator().manual_seed(seed)).to(b.dev)
    given = torch.zeros(2, dtype=torch.bool, device=b.dev)
    ts = timesteps(STEPS, SHIFT).to(b.dev)
    xs, guess, vcs, vus = [x[0]], [], [], []
    for i in range(STEPS):
        v = b.model(x.expand(2, -1, -1, -1, -1), ts[i].expand(2), both_txt, both_mask, given)
        v_c, v_u = v[:1], v[1:]
        vg = v_u + cfg * (v_c - v_u)
        guess.append((x - ts[i] * vg)[0])
        x = x - vg * (ts[i] - ts[i + 1])
        xs.append(x[0]); vcs.append(v_c[0]); vus.append(v_u[0])
    f16 = lambda L: torch.stack(L).half().cpu().numpy()
    np.savez_compressed(out / "tmp.npz", xs=f16(xs), guess=f16(guess), vc=f16(vcs), vu=f16(vus), t=ts.cpu().numpy())
    for s in DECODE_AT:
        z = guess[s - 1][None].float() * b.std + b.mean
        video = b.vae.decode(z).sample
        frames = ((video.clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()[:-1]
        yield s, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = HERE / "raw"
    out.mkdir(exist_ok=True)
    b = Bhaav(device=args.device)

    tok = []
    for p in PROMPTS:                     # what the text panel shows: T5's word pieces and a corner of its vectors
        ids = b.tok(p).input_ids
        emb, m = b.encode([p])
        n = int(m.sum())
        tok.append({"prompt": p, "pieces": b.tok.convert_ids_to_tokens(ids), "n_tokens": n,
                    "max_len": b.max_len, "dim": emb.shape[-1],
                    "corner": emb[0, :5, :5].round(decimals=2).tolist()})
    (out / "text.json").write_text(json.dumps({"pairs": [[k, 2 * i, 2 * i + 1] for i, (k, _, _) in enumerate(PAIRS)],
                                               "prompts": tok}, indent=1))

    for p, seed, cfg in grid():
        rid = run_id(p, seed, cfg)
        if (out / f"{rid}.npz").exists():
            continue
        t0 = time.time()
        for s, frames in one_run(b, PROMPTS[p], seed, cfg, out):
            to_strip(frames).save(out / f"{rid}_s{s}.jpg", quality=92)
        (out / "tmp.npz").rename(out / f"{rid}.npz")     # npz last: its presence marks the run finished
        print(f"{rid}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
