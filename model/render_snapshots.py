"""Decode the snapshot sample latents into GIFs you can look at.  (OUR code.)

For every snapshots/samples_stepNNNNNN.pt it writes:
  grid_stepNNNNNN.gif      the 8 sample stickers playing, 4 x 2, with a short label under each
And once for the whole run:
  progress.png             one row per snapshot, frame 5 of each of the 8 samples: watch quality improve down the rows

Usage (pod, after training):
  python render_snapshots.py --run /workspace/runs/r1 --cache /workspace/cache
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

TILE, LABEL_H = 192, 34
KIND = {"test_seen": "SEEN", "test_unseen": "UNSEEN", "image_to_sticker": "IMG->STICKER"}


def label_strip(lab):
    img = Image.new("RGB", (TILE, LABEL_H), "white")
    d = ImageDraw.Draw(img)
    d.text((3, 1), KIND[lab["kind"]], fill=(170, 30, 30) if lab["kind"] == "test_unseen" else (30, 30, 30))
    words, line, lines = lab["caption"].split(), "", []
    for w in words:                                        # wrap to two short lines
        if len(line) + len(w) > 34:
            lines.append(line); line = ""
        line += w + " "
    lines.append(line)
    for k, l in enumerate(lines[:2]):
        d.text((3, 12 + 10 * k), l.strip()[:36], fill=(80, 80, 80))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    from diffusers import AutoencoderKLWan

    snaps = Path(a.run) / "snapshots"
    vae_cfg = json.loads((Path(a.cache) / "latents" / "vae_config.json").read_text())
    mean = torch.tensor(vae_cfg["latents_mean"]).view(1, -1, 1, 1, 1).cuda()
    std = torch.tensor(vae_cfg["latents_std"]).view(1, -1, 1, 1, 1).cuda()
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae",
                                           torch_dtype=torch.float32).cuda().eval()
    rows = []
    for f in sorted(snaps.glob("samples_step*.pt")):
        s = torch.load(f)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            video = torch.cat([vae.decode(z[None].cuda() * std + mean).sample.float() for z in s["latents"]])
        clips = ((video.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu().numpy()  # [8,17,H,W,3]
        labels = [label_strip(l) for l in s["labels"]]
        frames = []
        for t in range(16):                                # 16 frames: frame 17 repeats frame 1
            grid = Image.new("RGB", (4 * TILE, 2 * (TILE + LABEL_H)), "white")
            for k, clip in enumerate(clips):
                x, y = (k % 4) * TILE, (k // 4) * (TILE + LABEL_H)
                grid.paste(Image.fromarray(clip[t]).resize((TILE, TILE)), (x, y))
                grid.paste(labels[k], (x, y + TILE))
            frames.append(grid)
        frames[0].save(snaps / f"grid_step{s['step']:06d}.gif", save_all=True, append_images=frames[1:],
                       duration=125, loop=0)
        rows.append((s["step"], [Image.fromarray(c[4]).resize((128, 128)) for c in clips]))
        print("rendered step", s["step"], flush=True)

    sheet = Image.new("RGB", (90 + 8 * 128, len(rows) * 128), "white")
    d = ImageDraw.Draw(sheet)
    for r, (step, tiles) in enumerate(rows):
        d.text((6, r * 128 + 58), f"step {step}", fill="black")
        for k, im in enumerate(tiles):
            sheet.paste(im, (90 + k * 128, r * 128))
    sheet.save(snaps / "progress.png")
    print("wrote", snaps / "progress.png")


if __name__ == "__main__":
    main()
