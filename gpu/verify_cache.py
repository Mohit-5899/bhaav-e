"""Check the 3b/3c caches before we trust them for training.  (OUR code.)

Checks:
  1. Latents: every id in all_ids.txt appears exactly once across the shards; shapes, dtype, all finite.
  2. Round trip from disk: decode 20 random cached latents with the VAE and compare to the original mp4.
     PSNR should match step 3a (~38 dB). If saving or loading scrambled anything, it would drop to ~10 dB.
  3. Text: every clip has its caption rows; row 0 is the empty caption; emb/mask shapes agree.

Usage on the pod:
    python verify_cache.py --clips /root/data/clips --ids /root/data/splits/all_ids.txt --cache /workspace/cache
"""
import argparse, json, random
from pathlib import Path

import av
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from safetensors import safe_open

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def read_clip(path):
    with av.open(str(path)) as f:
        return np.stack([fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)])


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    cache = Path(a.cache)
    want = Path(a.ids).read_text().split()

    # 1. latents: coverage and sanity
    where, seen = {}, []
    for shard in sorted((cache / "latents").glob("shard_*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            ids = json.loads(f.metadata()["ids"])
            t = f.get_tensor("latents")
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == (len(ids), 48, 5, 16, 16), (shard.name, t.shape)
        assert torch.isfinite(t.float()).all(), shard.name
        seen += ids
        where.update({c: (shard, k) for k, c in enumerate(ids)})
    dup = len(seen) - len(set(seen))
    missing = set(want) - set(seen)
    print(f"latents: {len(seen)} clips in {len(list((cache / 'latents').glob('shard_*')))} shards | "
          f"duplicates {dup} | missing {len(missing)} | extra {len(set(seen) - set(want))}")
    assert dup == 0 and not missing

    # 2. decode random cached latents and compare to the original videos
    paths = {p.stem: p for p in Path(a.clips).rglob("*.mp4")}
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    scores = []
    for c in random.Random(0).sample(want, a.n):
        shard, k = where[c]
        with safe_open(str(shard), "pt") as f:
            z = f.get_slice("latents")[k:k + 1].float().cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = vae.decode(z).sample
        rec = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()
        scores.append(psnr(read_clip(paths[c]), rec))
    print(f"decode-from-cache PSNR over {a.n} random clips: mean {np.mean(scores):.2f} dB, min {np.min(scores):.2f} dB")
    assert np.min(scores) > 25, "a cached latent does not decode to its own clip"

    # 3. text cache
    with safe_open(str(cache / "text" / "text.safetensors"), "pt") as f:
        emb, mask = f.get_tensor("emb"), f.get_tensor("mask")
    sentences = json.loads((cache / "text" / "captions.json").read_text())
    clip_rows = json.loads((cache / "text" / "clip_captions.json").read_text())
    assert emb.shape[:2] == mask.shape and emb.shape[0] == len(sentences) and sentences[0] == ""
    no_caps = [c for c in want if not clip_rows.get(c)]
    c = want[0]
    print(f"text: {emb.shape[0]} sentences x {emb.shape[1]} tokens x {emb.shape[2]} | clips without captions {len(no_caps)} | "
          f"example {c}: {sentences[clip_rows[c][0]]!r}")
    assert not no_caps
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
