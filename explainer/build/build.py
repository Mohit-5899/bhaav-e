"""Step 2 of the explainer build: turn the raw runs into the small files the page plays back.  (OUR code.)

1. Fit a linear map from the 48 latent channels to RGB (a "quick look" at a latent without the VAE).
   Pairs: each finished latent's 16 x 16 positions vs the decoded clip shrunk to 16 x 16, frame by frame.
   Latent frame 0 covers video frame 0; latent frame k >= 1 covers frames 4k-3 .. 4k (the VAE's 4x time squeeze).
2. Per run, one PNG of 16 px tiles (5 latent frames per row, one row per step):
     columns 0-4   the latent the sampler holds (quick look)
     columns 5-9   the model's guess of the finished clip (quick look)
     columns 10-14 where the caption changed the prediction: |v_caption - v_empty|, averaged over channels
   and one JPEG with the fully decoded guess at each checkpoint step (one row per checkpoint, 16 frames across).
3. For each prompt pair (same seed and guidance), a 2-D PCA of the two paths' 102 latents, for the compare chart.
   One PCA per pair: a global one is dominated by noise vs clip and draws the twins on top of each other.
4. A noising strip for the training section: (1 - t) * clip + t * noise at 11 values of t.
Writes ../assets/ and ../data.json.
Run: ../../.venv/bin/python build.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from precompute import CFGS, DECODE_AT, FULL_CFG, PAIRS, PROMPTS, SEEDS, SHIFT, STEPS, grid, run_id

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE.parent / "assets"
TILE, CLIP_PX = 16, 160
LABELS = ["Crying · tears", "Crying · no effect", "Furious · shaking", "Furious · pulsing",
          "Party · confetti", "Party · sparkles", "Cool · peach", "Cool · dark navy"]


def frames_of(strip):
    """JPEG strip of 16 frames side by side -> [16, H, W, 3] float in 0..1."""
    a = np.asarray(Image.open(strip).convert("RGB"), dtype=np.float32) / 255
    h = a.shape[0]
    return a.reshape(h, 16, h, 3).transpose(1, 0, 2, 3)


def shrink(frames, n=16):
    f, h = frames.shape[0], frames.shape[1]
    return frames.reshape(f, n, h // n, n, h // n, 3).mean((2, 4))


def video_to_latent_frames(small):
    """[16, 16, 16, 3] video frames -> [5, 16, 16, 3], averaged over the frames each latent frame covers."""
    groups = [[0]] + [[i for i in range(4 * k - 3, 4 * k + 1) if i < 16] for k in range(1, 5)]
    return np.stack([small[g].mean(0) for g in groups])


def fit_map(runs):
    X, Y = [], []
    for rid in runs:
        z = np.load(RAW / f"{rid}.npz")["xs"][-1].astype(np.float32)          # [48, 5, 16, 16]
        rgb = video_to_latent_frames(shrink(frames_of(RAW / f"{rid}_s{STEPS}.jpg")))
        X.append(z.reshape(48, -1).T)
        Y.append(rgb.reshape(-1, 3))
    X, Y = np.concatenate(X), np.concatenate(Y)
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)                                # [49, 3]
    pred = Xb @ W
    r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return W, [round(float(v), 3) for v in r2], len(X)


def quick_look(z, W):
    """[..., 48, 5, 16, 16] -> [..., 5, 16, 16, 3] in 0..1."""
    z = np.moveaxis(z.astype(np.float32), -4, -1)
    return np.clip(z @ W[:48] + W[48], 0, 1)


def tiles(rows):
    """rows: [R, 5, 16, 16, 3] -> image [R*16, 5*16, 3]."""
    r = rows.shape[0]
    return rows.transpose(0, 2, 1, 3, 4).reshape(r * TILE, 5 * TILE, 3)


def heat(d):
    """[50, 5, 16, 16] magnitudes -> orange heat tiles [50, 5, 16, 16, 3], each step scaled by its own 99th
    percentile, so the strip shows *where* the words act; the line chart shows *how much* (step 1 dwarfs the rest)."""
    v = np.clip(d / (np.percentile(d, 99, axis=(1, 2, 3), keepdims=True) + 1e-6), 0, 1)[..., None]
    lo, hi = np.array([0.06, 0.07, 0.10]), np.array([1.0, 0.55, 0.15])
    return lo + v * (hi - lo) + (v ** 3) * (1 - hi)                          # dark -> orange -> near white


def pca2(paths):
    """paths: {rid: [51, D]} -> 2-D coordinates, via the Gram matrix (51 x runs rows is small)."""
    keys = list(paths)
    X = np.concatenate([paths[k] for k in keys]).astype(np.float32)
    mu = X.mean(0)
    X -= mu
    G = X @ X.T
    val, vec = np.linalg.eigh(G)
    top = np.argsort(val)[::-1][:2]
    coords = vec[:, top] * np.sqrt(np.maximum(val[top], 0))
    share = [round(float(v), 3) for v in val[top] / val.sum()]
    out, i = {}, 0
    for k in keys:
        n = len(paths[k])
        out[k] = coords[i:i + n].round(2).tolist()
        i += n
    return out, share


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    runs = {run_id(*k): k for k in grid() if (RAW / f"{run_id(*k)}.npz").exists()}
    print(f"{len(runs)} runs")
    W, r2, n_px = fit_map(runs)
    print("quick-look map R^2 (R, G, B):", r2, "from", n_px, "positions")

    paths, meta = {}, {}
    for rid in runs:
        d = np.load(RAW / f"{rid}.npz")
        xs, guess, vc, vu = d["xs"], d["guess"], d["vc"], d["vu"]
        imp = np.abs(vc.astype(np.float32) - vu.astype(np.float32)).mean(1)   # [50, 5, 16, 16]
        a = tiles(quick_look(xs, W))
        b = np.concatenate([np.zeros((1, 5, 16, 16, 3)), quick_look(guess, W)])
        c = np.concatenate([np.zeros((1, 5, 16, 16, 3)), heat(imp)])
        sheet = np.concatenate([a, tiles(b), tiles(c)], 1)
        Image.fromarray((sheet * 255).round().astype(np.uint8)).save(OUT / "runs" / f"{rid}.png", optimize=True)

        rows, decoded = [], DECODE_AT
        for s in decoded:
            f = Image.open(RAW / f"{rid}_s{s}.jpg").convert("RGB")
            rows.append(np.asarray(f.resize((16 * CLIP_PX, CLIP_PX), Image.LANCZOS)))
        Image.fromarray(np.concatenate(rows)).save(OUT / "runs" / f"{rid}.jpg", quality=82, optimize=True)

        paths[rid] = xs.reshape(len(xs), -1)
        meta[rid] = {"impact": imp.mean((1, 2, 3)).round(4).tolist(), "decoded": decoded}

    for _, (p, seed, cfg) in runs.items():
        if p % 2:
            continue
        a, b = run_id(p, seed, cfg), run_id(p + 1, seed, cfg)
        if b in runs:
            traj, share = pca2({a: paths[a], b: paths[b]})
            for k in (a, b):
                meta[k]["traj"], meta[k]["trajShare"] = traj[k], round(sum(share), 4)

    # training strip: one finished clip, one fresh noise, mixed at 11 values of t
    x0 = np.load(RAW / f"{run_id(0, 0, 3.0)}.npz")["xs"][-1].astype(np.float32)
    eps = np.random.default_rng(123).standard_normal(x0.shape).astype(np.float32)
    tt = np.linspace(0, 1, 11)
    mix = np.stack([(1 - t) * x0 + t * eps for t in tt])
    Image.fromarray((tiles(quick_look(mix, W)) * 255).round().astype(np.uint8)).save(OUT / "train.png", optimize=True)

    text = json.loads((RAW / "text.json").read_text())
    t = (lambda u: SHIFT * u / (1 + (SHIFT - 1) * u))(np.linspace(1, 0, STEPS + 1))
    data = {
        "prompts": [dict(p, label=LABELS[i]) for i, p in enumerate(text["prompts"])],
        "pairs": [{"kind": k, "a": 2 * i, "b": 2 * i + 1} for i, (k, _, _) in enumerate(PAIRS)],
        "seeds": SEEDS, "cfgs": CFGS, "steps": STEPS, "shift": SHIFT, "t": t.round(4).tolist(),
        "decodeAt": DECODE_AT, "fullCfg": FULL_CFG, "tile": TILE, "clipPx": CLIP_PX,
        "quickLook": {"r2": r2, "positions": n_px},
        "train": {"t": tt.round(2).tolist(), "run": run_id(0, 0, 3.0)},
        "runs": meta,
    }
    (HERE.parent / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {len(runs)} runs, assets {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
