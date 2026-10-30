"""Reads the step 3b/3c caches for training.  (OUR code.)

Why everything is held in memory: the latents are 9.2 GB (bf16) and the pod has 123 GB of RAM, so we load them once
and pick random batches by index. No disk reads during training, so the GPU never waits.
The caption vectors (1.4 GB) go straight onto the GPU, since every step reads them.

Normalising: VAE latents have a different average and spread per channel. Training works best when every number is
roughly "average 0, spread 1", so we subtract each channel's mean and divide by its std (from the VAE's own config).
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open


class Cache:
    def __init__(self, cache_dir, split_file, device="cpu"):
        cache = Path(cache_dir)
        want = set(Path(split_file).read_text().split())

        lat, ids = [], []
        for shard in sorted((cache / "latents").glob("shard_*.safetensors")):
            with safe_open(str(shard), "pt") as f:
                shard_ids = json.loads(f.metadata()["ids"])
                keep = [k for k, c in enumerate(shard_ids) if c in want]
                if keep:
                    lat.append(f.get_tensor("latents")[keep])
                    ids += [shard_ids[k] for k in keep]
        assert len(ids) == len(want), f"cache has {len(ids)} of {len(want)} clips in {split_file}"
        self.ids, self.latents = ids, torch.cat(lat)                       # bf16 [N, 48, 5, 16, 16], CPU

        vae = json.loads((cache / "latents" / "vae_config.json").read_text())
        self.mean = torch.tensor(vae["latents_mean"], device=device).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(vae["latents_std"], device=device).view(1, -1, 1, 1, 1)

        with safe_open(str(cache / "text" / "text.safetensors"), "pt") as f:
            self.emb = f.get_tensor("emb").to(device)                         # bf16 [S, L, 768]
            self.mask = f.get_tensor("mask").to(device)                       # uint8 [S, L], 1 = real word
        rows = json.loads((cache / "text" / "clip_captions.json").read_text())
        self.rows = [rows[c] for c in ids]                                    # each clip's caption rows
        self.device = device

    def __len__(self):
        return len(self.ids)

    def normalize(self, z):
        return (z.float() - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def batch(self, idx, gen, caption_drop=0.0):
        """Clips at positions idx -> (x0, text, mask). Each clip gets one of its caption lines at random;
        with probability caption_drop it gets the empty caption (row 0) instead, for classifier-free guidance."""
        x0 = self.normalize(self.latents[idx].to(self.device, non_blocking=True))
        pick = torch.rand(len(idx), generator=gen)
        drop = torch.rand(len(idx), generator=gen) < caption_drop
        rows = [0 if d else r[int(u * len(r))] for d, r, u in zip(drop.tolist(), (self.rows[i] for i in idx), pick.tolist())]
        rows = torch.tensor(rows, device=self.device)
        return x0, self.emb[rows].float(), self.mask[rows]   # float: stored as bf16, autocast picks the GPU precision
