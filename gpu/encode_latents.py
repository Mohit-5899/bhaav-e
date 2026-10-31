"""Step 3b: encode all 75,001 clips once with the frozen Wan2.2 VAE -> latent cache.  (OUR code. Untested sketch: not run yet.)

Why: the VAE never changes, so encoding each clip on every training step would redo the same work
~187 times (see docs/how-training-works.md, section 7). Encode once, train on the saved latents.

Usage on the pod:
    python encode_latents.py --clips /workspace/clips --ids /workspace/splits/all_ids.txt --out /workspace/cache/latents
Output: shard_00000.safetensors ... each holds up to 4,096 clips:
    "latents": bf16 [n, 48, 5, 16, 16]   (posterior MEAN, NOT normalised; normalise at train time
                                           with vae.config.latents_mean / latents_std)
    "ids":     the clip ids, in the same order (stored in the shard's metadata)
Size check: 75,001 x 61,440 numbers x 2 bytes = 9.22 GB.
Resumable: shards that already exist are skipped, so a crash or a stopped pod loses at most one shard.
"""
import argparse, json
from pathlib import Path

import av
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SHARD = 4096


class Clips(Dataset):
    """Decodes mp4s on CPU worker processes so the GPU never waits for video decoding."""
    def __init__(self, ids, paths):
        self.ids, self.paths = ids, paths

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        with av.open(str(self.paths[self.ids[i]])) as f:
            frames = np.stack([fr.to_ndarray(format="rgb24") for fr in f.decode(video=0)])
        assert frames.shape == (17, 256, 256, 3), (self.ids[i], frames.shape)
        return torch.from_numpy(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--ids", required=True, help="all ids to encode: train+val+test_seen+test_unseen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)  # set from step 3a: largest batch that fits
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    paths = {p.stem: p for p in Path(a.clips).rglob("*.mp4")}
    ids = sorted(Path(a.ids).read_text().split())
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    (out / "vae_config.json").write_text(json.dumps(dict(vae.config), default=str))  # keep latents_mean/std with the cache

    for s in range(0, len(ids), SHARD):
        dst = out / f"shard_{s // SHARD:05d}.safetensors"
        if dst.exists():
            continue
        shard_ids = ids[s:s + SHARD]
        loader = DataLoader(Clips(shard_ids, paths), batch_size=a.batch, num_workers=a.workers, pin_memory=True)
        parts = []
        for u8 in loader:
            x = u8.cuda(non_blocking=True).permute(0, 4, 1, 2, 3).float() / 127.5 - 1.0  # [B,3,17,256,256] in [-1,1]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = vae.encode(x).latent_dist.mean
            parts.append(z.to(torch.bfloat16).cpu())
        lat = torch.cat(parts)
        assert lat.shape == (len(shard_ids), 48, 5, 16, 16) and torch.isfinite(lat.float()).all()
        tmp = dst.with_suffix(".tmp")
        save_file({"latents": lat.contiguous()}, str(tmp), metadata={"ids": json.dumps(shard_ids)})
        tmp.rename(dst)  # only a complete file gets the final name, so resume never trusts a half-written shard
        print(f"{dst.name}: {len(shard_ids)} clips, done {min(s + SHARD, len(ids))}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
