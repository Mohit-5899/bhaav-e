"""Step 3a: Wan2.2 VAE round-trip test on ~200 Bhaav clips.  (OUR code. Untested sketch: not run yet.)

Question it answers: can the frozen Wan2.2 VAE rebuild our stickers well enough?
If not, no generator trained on its latents can make good stickers (see how-training-works.md, 3e).

Usage on the pod:
    python vae_roundtrip.py --clips /workspace/clips --split /workspace/splits/val.txt --n 200 --out /workspace/roundtrip
Outputs:
    roundtrip/results.csv        one row per clip: id, category, psnr, psnr_bf16_latent, ssim
    roundtrip/summary.txt        mean / worst PSNR per category, clips per second (to forecast step 3b)
    roundtrip/gifs/<id>.gif      original | rebuilt | 4x error map, side by side
"""
import argparse, csv, random, time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan
from skimage.metrics import structural_similarity

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"  # the 48-channel, 16x spatial / 4x time Wan2.2 VAE (Apache-2.0)


def read_clip(path):
    """mp4 -> uint8 array [17, 256, 256, 3]."""
    with av.open(str(path)) as f:
        frames = [fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)]
    assert len(frames) == 17, (path, len(frames))
    return np.stack(frames)


def to_tensor(clips_u8):
    """[B,17,H,W,3] uint8 -> [B,3,17,H,W] float in [-1, 1], the range the VAE was trained on."""
    x = torch.from_numpy(clips_u8).permute(0, 4, 1, 2, 3).float()
    return x / 127.5 - 1.0


def to_u8(x):
    """[B,3,17,H,W] in [-1,1] -> [B,17,H,W,3] uint8."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.permute(0, 2, 3, 4, 1).cpu().numpy()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def save_gif(orig, rec, path):
    err = np.clip(np.abs(orig.astype(int) - rec.astype(int)) * 4, 0, 255).astype(np.uint8)
    frames = [Image.fromarray(np.concatenate([o, r, e], axis=1)) for o, r, e in zip(orig, rec, err)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=125, loop=0)  # 8 fps


def pick_clips(split_file, n, seed=0):
    """Stratified sample: the same number from each subcategory, so rare ones (e.g. ui/toggle) are tested too."""
    by_sub = defaultdict(list)
    for cid in Path(split_file).read_text().split():
        by_sub[cid.rsplit("-", 1)[0]].append(cid)
    rng = random.Random(seed)
    per = max(1, n // len(by_sub))
    return [c for ids in by_sub.values() for c in rng.sample(ids, min(per, len(ids)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out); (out / "gifs").mkdir(parents=True, exist_ok=True)
    paths = {p.stem: p for p in Path(a.clips).rglob("*.mp4")}
    ids = pick_clips(a.split, a.n)

    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).cuda().eval()

    rows, t_encode, n_done = [], 0.0, 0
    for i in range(0, len(ids), a.batch):
        bid = ids[i:i + a.batch]
        orig = np.stack([read_clip(paths[c]) for c in bid])
        x = to_tensor(orig).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize(); t0 = time.time()
            z = vae.encode(x).latent_dist.mean          # [B,48,5,16,16]; the mean, not a random sample
            torch.cuda.synchronize(); t_encode += time.time() - t0
            rec = to_u8(vae.decode(z.float()).sample)
            # Same latent rounded to bf16, which is how the cache will store it (step 3b).
            rec_bf16 = to_u8(vae.decode(z.to(torch.bfloat16).float()).sample)
        assert z.shape[1:] == (48, 5, 16, 16), z.shape
        n_done += len(bid)
        for c, o, r, rb in zip(bid, orig, rec, rec_bf16):
            ssim = np.mean([structural_similarity(of, rf, channel_axis=2) for of, rf in zip(o, r)])
            rows.append(dict(id=c, category=next(k for k in ("animated-emoji", "reactions", "ui") if c.startswith(k + "-")), psnr=psnr(o, r), psnr_bf16_latent=psnr(o, rb), ssim=ssim))
            save_gif(o, r, out / "gifs" / f"{c}.gif")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    lines = [f"clips: {n_done}   encode speed: {n_done / t_encode:.1f} clips/s (encode only, bf16 autocast)"]
    for cat in sorted({r['category'] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if cat in ("ALL", r["category"])]
        p = np.array([r["psnr"] for r in rs])
        worst = min(rs, key=lambda r: r["psnr"])
        lines.append(f"{cat:15s} n={len(rs):3d}  PSNR mean {p.mean():.2f}  p10 {np.percentile(p, 10):.2f}  "
                     f"worst {worst['psnr']:.2f} ({worst['id']})  "
                     f"bf16-latent loss {np.mean([r['psnr'] - r['psnr_bf16_latent'] for r in rs]):.3f} dB  "
                     f"SSIM {np.mean([r['ssim'] for r in rs]):.4f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
