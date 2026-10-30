"""Train Bhaav Tiny's VideoDiT with rectified flow on the cached latents.  (OUR code.)

One training step (how-training-works.md, section 9):
  1. take a batch of clean latents x0 and one caption line each (10% of captions replaced by the empty caption)
  2. pick a noise level t per clip (logit-normal: most t near the middle, where learning is hardest)
  3. mix clean and noise on a straight line:  x_t = (1 - t) * x0 + t * noise
  4. 25% of clips: frame 0 is given clean (image -> sticker), tagged, and not scored
  5. the model guesses the velocity; the right answer is v = noise - x0; loss = mean squared error
  6. backprop, clip the gradient size to 1.0, optimiser step (Muon + AdamW), update the slow EMA copy

Usage (pod):
  python train.py --cache /workspace/cache --splits /root/data/splits --out /workspace/runs/r1 --steps 20000
Overfit test (proves the code can learn; noise and t are fixed so the loss can reach ~0):
  python train.py ... --overfit 8 --steps 300 --size tiny
"""
import argparse, copy, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import Cache
from muon import make_optimizers
from videodit import SIZES, VideoDiT, count_params


def lr_factor(step, total, warmup, decay_frac):
    """Warm up linearly, hold, then decay linearly to 0 over the last decay_frac of training."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = total * (1 - decay_frac)
    return 1.0 if step < decay_start else max(0.0, (total - step) / (total - decay_start))


def flow_loss(model, x0, txt, mask, t, noise, first_given):
    """Rectified-flow loss; frame 0 is excluded from the score where it was given clean."""
    xt = (1 - t)[:, None, None, None, None] * x0 + t[:, None, None, None, None] * noise
    xt = torch.where(first_given[:, None, None, None, None] & (torch.arange(x0.shape[2], device=x0.device) == 0)
                     [None, None, :, None, None], x0, xt)
    v = model(xt, t, txt, mask, first_given)
    err = (v.float() - (noise - x0)).pow(2)
    keep = torch.ones_like(err)
    keep[first_given, :, 0] = 0
    return (err * keep).sum() / keep.sum()


@torch.no_grad()
def val_loss(model, data, device, n=512, seed=1234):
    """Same clips, same noise, same t every time, so changes in this number are real, not luck."""
    g = torch.Generator().manual_seed(seed)
    idx = list(range(min(n, len(data))))
    tot, cnt = 0.0, 0
    for i in range(0, len(idx), 64):
        b = idx[i:i + 64]
        x0, txt, mask = data.batch(b, g)
        t = torch.linspace(0.05, 0.95, len(b))[torch.randperm(len(b), generator=g)].to(device)
        noise = torch.randn(x0.shape, generator=g).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss = flow_loss(model, x0, txt, mask, t, noise, torch.zeros(len(b), dtype=torch.bool, device=device))
        tot += loss.item() * len(b); cnt += len(b)
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="s210", choices=SIZES)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay_frac", type=float, default=0.25)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--adam_lr", type=float, default=2e-4)
    ap.add_argument("--no_muon", action="store_true")
    ap.add_argument("--caption_drop", type=float, default=0.1)
    ap.add_argument("--p_first", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--snap_every", type=int, default=5000, help="keep a snapshot + sample stickers every N steps")
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed clips with fixed noise and t")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev, out = a.device, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = dev.startswith("cuda")
    train = Cache(a.cache, Path(a.splits) / "train.txt", dev)
    val = Cache(a.cache, Path(a.splits) / "val.txt", dev)

    model = VideoDiT(SIZES[a.size]).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opts = make_optimizers(model, a.muon_lr, a.adam_lr, use_muon=not a.no_muon)
    for o in opts:
        for grp in o.param_groups:
            grp["base_lr"] = grp["lr"]
    step = 0
    ckpt = out / "last.pt"
    if ckpt.exists():                                       # resume after a crash or a stopped pod
        s = torch.load(ckpt, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"])
        for o, so in zip(opts, s["opts"]):
            o.load_state_dict(so)
        step = s["step"]
        print(f"resumed at step {step}")
    fwd = torch.compile(model) if (a.compile and cuda) else model
    sample_set = None if a.overfit else build_sample_set(a, dev)
    print(json.dumps(dict(params_M=round(count_params(model) / 1e6, 2), train=len(train), val=len(val), **vars(a))))

    g = torch.Generator().manual_seed(a.seed + step)
    if a.overfit:                                           # fixed clips, captions, noise and t
        fixed_idx = list(range(a.overfit))
        fx0, ftxt, fmask = train.batch(fixed_idx, torch.Generator().manual_seed(0))
        fnoise = torch.randn(fx0.shape, generator=torch.Generator().manual_seed(1)).to(dev)
        ft = torch.sigmoid(torch.randn(a.overfit, generator=torch.Generator().manual_seed(2))).to(dev)

    log = open(out / "log.jsonl", "a")
    t_last, n_since = time.time(), 0
    while step < a.steps:
        f = lr_factor(step, a.steps, a.warmup if not a.overfit else 1, a.decay_frac)
        for o in opts:
            for grp in o.param_groups:
                grp["lr"] = grp["base_lr"] * f

        if a.overfit:
            x0, txt, mask, noise, t = fx0, ftxt, fmask, fnoise, ft
            first = torch.zeros(a.overfit, dtype=torch.bool, device=dev)
        else:
            idx = torch.randint(0, len(train), (a.batch,), generator=g).tolist()
            x0, txt, mask = train.batch(idx, g, a.caption_drop)
            t = torch.sigmoid(torch.randn(a.batch, generator=g)).to(dev)
            noise = torch.randn(x0.shape, generator=g).to(dev)
            first = (torch.rand(a.batch, generator=g) < a.p_first).to(dev)

        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
            loss = flow_loss(fwd, x0, txt, mask, t, noise, first)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        d = min(a.ema, (1 + step) / (10 + step))            # EMA warms up: early on, follow the model closely
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        n_since += len(x0)

        if step % a.log_every == 0 or step == a.steps:
            now = time.time()
            row = dict(step=step, loss=round(loss.item(), 5), gnorm=round(gnorm.item(), 3), lr_f=round(f, 4),
                       clips_per_s=round(n_since / (now - t_last), 1),
                       mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 1) if cuda else None)
            t_last, n_since = now, 0
            if not a.overfit and (step % a.val_every == 0 or step == a.steps):
                row["val"] = round(val_loss(ema, val, dev), 5)
            print(json.dumps(row), flush=True); log.write(json.dumps(row) + "\n"); log.flush()

        if step % a.ckpt_every == 0 or step == a.steps:
            tmp = out / "last.tmp"
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opts=[o.state_dict() for o in opts],
                            step=step, size=a.size), tmp)
            tmp.rename(ckpt)                                # a half-written checkpoint never replaces a good one

        if not a.overfit and (step % a.snap_every == 0 or step == a.steps):
            snapshot(ema, a, step, out, dev, cuda, sample_set)


SAMPLE_SPLITS = (("test_seen", 3), ("test_unseen", 3), ("image_to_sticker", 2))


def build_sample_set(a, dev):
    """The same 8 prompts at every snapshot, so snapshots can be compared side by side:
    3 familiar combinations (test_seen), 3 never-seen combinations (test_unseen), 2 image -> sticker (a given frame 0).
    Built once at the start; reading the test clips again at every snapshot would rescan the whole cache."""
    from safetensors import safe_open
    caps = json.loads((Path(a.cache) / "text" / "captions.json").read_text())
    items = []
    for split, n in SAMPLE_SPLITS:
        src = Cache(a.cache, Path(a.splits) / ("test_seen.txt" if split == "image_to_sticker" else f"{split}.txt"), dev)
        offset = 7 if split == "image_to_sticker" else 0                 # different clips from the text-only ones
        for k in range(n):
            i = (offset + k * len(src) // n) % len(src)                  # spread over the split: different subjects
            row = src.rows[i][0]
            first = src.normalize(src.latents[i:i + 1].to(dev))[0, :, 0] if split == "image_to_sticker" else None
            items.append(dict(txt=src.emb[row].float(), mask=src.mask[row], first=first,
                              label=dict(kind=split, clip=src.ids[i], caption=caps[row])))
        del src
    with safe_open(str(Path(a.cache) / "text" / "text.safetensors"), "pt") as f:   # row 0 = the empty caption
        empty = (f.get_slice("emb")[0:1].float().to(dev), f.get_slice("mask")[0:1].to(dev))
    return items, empty


def snapshot(ema, a, step, out, dev, cuda, sample_set):
    """Keep this moment's EMA weights (bf16, about 425 MB) and the latents of the 8 sample stickers.
    Only latents are saved; render_snapshots.py decodes them with the VAE later (keeps GPU memory free for training)."""
    from sample import sample
    items, (empty_txt, empty_mask) = sample_set
    snaps = out / "snapshots"; snaps.mkdir(exist_ok=True)
    torch.save(dict(ema={k: v.to(torch.bfloat16) for k, v in ema.state_dict().items()}, step=step, size=a.size),
               snaps / f"ema_step{step:06d}.pt")
    latents = []
    with torch.no_grad(), torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
        for it in items:
            first = None if it["first"] is None else it["first"][None]
            z = sample(ema, it["txt"][None], it["mask"][None], empty_txt, empty_mask, first=first, seed=0)
            latents.append(z.float().cpu())
    torch.save(dict(latents=torch.cat(latents), labels=[it["label"] for it in items], step=step),
               snaps / f"samples_step{step:06d}.pt")
    print(json.dumps(dict(snapshot=step, samples=len(items))), flush=True)


if __name__ == "__main__":
    main()
