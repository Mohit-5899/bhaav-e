"""Seen vs unseen evaluation of a Bhaav Tiny run.  (OUR code.)

The question: did the model learn *parts* it can mix (subject, motion, effect, background, accent), or did it only
memorise the combinations it trained on? test_unseen holds combinations that never appear in training
(data/splits/heldout.json), so if the model only memorised, it should do clearly worse there than on test_seen.

Part A, loss (cheap, every snapshot):
  The training loss, measured on every test clip with fixed noise, t and caption line, for each of the 8 EMA snapshots.
  Same number as "val" in the log, so seen / unseen / val sit on one scale. The gap unseen - seen is the memorising signal.
  Limit: a low loss means "good at removing noise", not "the finished sticker follows the caption".

Part B, does the finished sticker follow the caption? (final snapshot):
  1. Make N stickers from test_seen captions and N from test_unseen captions (50 steps, CFG 5, as in sampling).
  2. Read each attribute back out of the pixels with simple checkers:
       background  colour of the outer border ring, nearest named background colour
       subject     CLIP image embedding (5 frames averaged), nearest neighbours among real training clips
       effect      the same CLIP embedding minus the average look of its subject and background, nearest neighbours
       motion      hand-made movement numbers (how the shape's centre, box, size and tilt change over frames), nearest neighbours
       accent (UI) average colour of the foreground pixels, nearest neighbours among real UI clips
  3. Run the same checkers on the *real* clips of the same captions (VAE round-trip, so both go through the same decoder).
     The real score is the checker's own ceiling: if the checker gets motion right 80% of the time on real clips, a model
     score of 78% is near perfect, not "22% wrong".
  4. Paired (v2): count the model's errors only on clips whose REAL version the checker reads correctly, so checker
     mistakes are not blamed on the model. Every clip's readings are saved in report.json ("per_clip").

Usage (pod):
  python evaluate.py --run /workspace/runs/r1 --cache /root/cache --splits /root/data/splits \
      --meta /workspace/meta.jsonl --out /workspace/runs/r1/eval
"""
import argparse, gc, json, math, os, random, resource, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from data import Cache
import effect_check
from motion_check import classify as motion_classify
from sample import sample
from videodit import SIZES, VideoDiT

ATTRS = ("subject", "motion", "effect", "background", "accent")


def labels(m):
    """What the caption asks for. Motion is not scored on UI icons (v3): they animate on their own (spinner turning,
    progress filling), and on real UI clips no checker we built could tell their extra still / float / pulse apart."""
    return dict(subject=f"{m['category']}/{m['subcategory']}", motion=None if m["category"] == "ui" else m["motion"],
                effect=m["effect"], background=m["background_name"], accent=m.get("accent"))


def load_ema(path, dev):
    s = torch.load(path, map_location=dev)
    model = VideoDiT(SIZES[s["size"]]).to(dev).eval()
    model.load_state_dict({k: v.float() for k, v in s["ema"].items()})
    return model, s["step"]


# ---------------- Part A: loss on every test clip ----------------

@torch.no_grad()
def per_clip_loss(model, data, dev, bs=256, seed=1234):
    """Loss per clip, with noise, t and caption line fixed by the clip's position, so every snapshot sees the same task."""
    g = torch.Generator().manual_seed(seed)
    n = len(data)
    t_all = torch.linspace(0.05, 0.95, n)[torch.randperm(n, generator=g)]
    out = torch.empty(n)
    for i in range(0, n, bs):
        idx = list(range(i, min(n, i + bs)))
        x0 = data.normalize(data.latents[idx].to(dev))
        rows = torch.tensor([data.rows[k][0] for k in idx], device=dev)
        txt, mask = data.emb[rows].float(), data.mask[rows]
        noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed + i)).to(dev)
        t = t_all[idx].to(dev)
        xt = (1 - t)[:, None, None, None, None] * x0 + t[:, None, None, None, None] * noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(xt, t, txt, mask, torch.zeros(len(idx), dtype=torch.bool, device=dev))
        out[i:i + len(idx)] = (v.float() - (noise - x0)).pow(2).mean(dim=(1, 2, 3, 4)).cpu()
    return out, t_all


# ---------------- Part B: generate, decode, read attributes back ----------------

def pick_spread(ids, meta, n, seed=0):
    """n clips spread over as many different combinations as possible (round-robin over subject x motion x effect x accent)."""
    groups = defaultdict(list)
    for c in ids:
        m = meta[c]
        groups[(m["category"], m["subcategory"], m["motion"], m["effect"], m.get("accent"))].append(c)
    rnd = random.Random(seed)
    keys = sorted(groups)
    rnd.shuffle(keys)
    for k in keys:
        rnd.shuffle(groups[k])
    picked, r = [], 0
    while len(picked) < n and any(len(groups[k]) > r for k in keys):
        for k in keys:
            if len(groups[k]) > r and len(picked) < n:
                picked.append(groups[k][r])
        r += 1
    return picked


@torch.no_grad()
def decode(vae, z_norm, cache, bs=8):
    """Normalised latents [N, 48, 5, 16, 16] -> uint8 video [N, 17, 256, 256, 3] on CPU."""
    out = []
    for i in range(0, len(z_norm), bs):
        z = z_norm[i:i + bs].cuda().float() * cache.std.cuda() + cache.mean.cuda()   # cache may live on the CPU
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = vae.decode(z).sample.float()
        out.append(((v.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu())
    return torch.cat(out)


def border_colour(clip):
    """Median colour of the outer 8-pixel ring over all frames: the background."""
    c = clip.float()
    ring = torch.cat([c[:, :8].reshape(-1, 3), c[:, -8:].reshape(-1, 3), c[:, :, :8].reshape(-1, 3), c[:, :, -8:].reshape(-1, 3)])
    return ring.median(dim=0).values


def extent(profile, lo=0.1, hi=0.9):
    """Where the 10% and 90% points of the foreground fall along one axis. Scattered particles (sparkles, confetti) are
    only a few pixels, so they barely move these points: this follows the main object, unlike the plain centre."""
    cdf = profile.cumsum(-1) / profile.sum(-1, keepdim=True).clamp(min=1)
    return (cdf < lo).sum(-1).float(), (cdf < hi).sum(-1).float()


def motion_and_colour(clip, bg):
    """Movement numbers from the foreground mask (pixels far from the background colour), per frame:
    centre x / y, size, tilt angle, brightness, and the main object's box (10%-90% extents). Then how much each changes
    over the 17 frames, how fast, and the biggest swing. Also returns the average foreground colour (for the UI accent).
    v2: runs on the GPU (v1 took 25 minutes on the CPU) and adds the box, which separates pulse (box grows) from float
    (box moves) better than the centre, which sparkles drag around."""
    c = clip.cuda().float()                                            # [17, H, W, 3]
    bg = bg.cuda()
    T, H, W, _ = c.shape
    fg = ((c - bg).norm(dim=-1) > 40).float()                          # [17, H, W]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32, device="cuda"),
                            torch.arange(W, dtype=torch.float32, device="cuda"), indexing="ij")
    area = fg.sum(dim=(1, 2)).clamp(min=1)
    cx = (fg * xs).sum(dim=(1, 2)) / area
    cy = (fg * ys).sum(dim=(1, 2)) / area
    dx, dy = xs - cx[:, None, None], ys - cy[:, None, None]
    mxx, myy, mxy = [(fg * a).sum(dim=(1, 2)) / area for a in (dx * dx, dy * dy, dx * dy)]
    angle = 0.5 * torch.atan2(2 * mxy, mxx - myy)
    bright = (c.mean(-1) * fg).sum(dim=(1, 2)) / area
    rel_area = area / area.mean()
    x0, x1 = extent(fg.sum(1))
    y0, y1 = extent(fg.sum(2))
    box_w, box_h = x1 - x0, y1 - y0
    series = (cx, cy, rel_area * 100, torch.rad2deg(angle), bright,
              (x0 + x1) / 2, (y0 + y1) / 2, box_w / box_w.mean() * 100, box_h / box_h.mean() * 100)
    feats = []
    for s in series:
        feats += [s.std(), s.diff().abs().mean(), s.max() - s.min()]   # how much it moves, how fast, biggest swing
    feats.append((c[1:] - c[:-1]).abs().mean())                        # overall pixel change between frames
    col = (c * fg[..., None]).sum(dim=(0, 1, 2)) / fg.sum().clamp(min=1)
    return torch.stack(feats).cpu().tolist(), col.cpu().tolist()


@torch.no_grad()
def clip_embed(model, proc, videos, bs=64):
    """CLIP image embedding of frames 0, 4, 8, 12, 16, averaged and unit length: what is in the picture."""
    frames = videos[:, [0, 4, 8, 12, 16]].reshape(-1, *videos.shape[2:]).numpy()
    embs = []
    for i in range(0, len(frames), bs * 5):
        inp = proc(images=list(frames[i:i + bs * 5]), return_tensors="pt").to("cuda")
        pooled = model.vision_model(pixel_values=inp["pixel_values"]).pooler_output   # same in every transformers version
        e = F.normalize(model.visual_projection(pooled).float(), dim=-1)
        embs.append(e.cpu())
    e = torch.cat(embs).view(len(videos), 5, -1).mean(1)
    return F.normalize(e, dim=-1)


def features(videos, clip_model, proc):
    bgs = torch.stack([border_colour(v.cuda()).cpu() for v in videos])
    mc = [motion_and_colour(v, b) for v, b in zip(videos, bgs)]
    return dict(bg=bgs, motion=torch.tensor([m for m, _ in mc]), colour=torch.tensor([c for _, c in mc]),
                clip=clip_embed(clip_model, proc, videos))


def knn(query, pool, pool_labels, k=10, cosine=False):
    """Vote among the k most similar real training clips."""
    if cosine:
        sim = query @ pool.T
    else:
        sim = -torch.cdist(query, pool)
    top = sim.topk(min(k, len(pool)), dim=1).indices
    return [Counter(pool_labels[j] for j in row.tolist()).most_common(1)[0][0] for row in top]


def residual(emb, subj, bgs, means):
    """Take out what the subject and the background look like on average, leaving mostly the effect.
    Why (v2): the v1 effect checker voted among whole-picture neighbours, so a bell found other bells and copied their
    usual effect; on unseen combinations it was right only 86% of the time even on real clips."""
    ms, mb, mg = means
    r = emb - torch.stack([ms[s] for s in subj]) - torch.stack([mb[b] for b in bgs]) + mg
    return F.normalize(r, dim=-1)


def group_means(emb, keys):
    out = {}
    for k in set(keys):
        out[k] = emb[[i for i, x in enumerate(keys) if x == k]].mean(0)
    return out


def _effect_features(args):
    video, bg = args
    return effect_check.features(video, bg)


def effect_features(videos, bgs, workers=None):
    """Effect v4 features for many clips, in parallel on the CPU (about 0.2 s per clip on one core)."""
    import multiprocessing as mp
    jobs = [(v.numpy(), b.tolist()) for v, b in zip(videos, bgs)]
    with mp.get_context("fork").Pool(workers or min(8, os.cpu_count())) as p:
        return np.stack(p.map(_effect_features, jobs, chunksize=4))


def predict(f, pool, pool_lab, bg_palette, videos, allowed, effect_v4=None):
    names = list(bg_palette)
    pal = torch.tensor([bg_palette[n] for n in names], dtype=torch.float32)
    pred = dict(background=[names[i] for i in torch.cdist(f["bg"].cpu(), pal).argmin(1).tolist()])
    pred["subject"] = knn(f["clip"], pool["clip"], pool_lab["subject"], cosine=True)
    means = (group_means(pool["clip"], pool_lab["subject"]), group_means(pool["clip"], pool_lab["background"]),
             pool["clip"].mean(0))
    pool_res = residual(pool["clip"], pool_lab["subject"], pool_lab["background"], means)
    q_res = residual(f["clip"], pred["subject"], pred["background"], means)   # the checker's own reading, not the caption
    pred["effect"] = knn(q_res, pool_res, pool_lab["effect"], cosine=True)
    if effect_v4 is not None:
        # effect v4 (effect_check.py): read the particles the generator draws, compared with real clips of the same
        # subject. The CLIP reading is kept as effect_v3 for comparison but no longer scored.
        pred["effect_v3"] = pred["effect"]
        ef = effect_features(videos, f["bg"])
        pred["effect"] = [effect_v4.predict(x, s) for x, s in zip(ef, pred["subject"])]
    # motion v3: compare with the generator's own formulas (motion_check.py), among the motions this subject can have
    pred["motion"] = [None if subj.startswith("ui/") else motion_classify(v.numpy(), bg.numpy(), allowed.get(subj))
                      for v, bg, subj in zip(videos, f["bg"], pred["subject"])]
    ui = [i for i, s in enumerate(pool_lab["subject"]) if s.startswith("ui/")]
    pred["accent"] = knn(f["colour"], pool["colour"][ui], [pool_lab["accent"][i] for i in ui], k=5)
    return pred


def wilson(k, n, z=1.96):
    """95% range for a success rate measured on n samples (wider when n is small)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(mid - half, 3), round(mid + half, 3))


def correct(pred, truth):
    """Per clip and attribute: True / False, or None where the attribute does not apply (accent outside UI).
    'all' = every attribute that applies is right."""
    ok = {a: [None if t[a] is None else pred[a][i] == t[a] for i, t in enumerate(truth)] for a in ATTRS}
    ok["all"] = [all(ok[a][i] is not False for a in ATTRS) for i in range(len(truth))]
    return ok


def score(ok):
    res = {}
    for a, v in ok.items():
        v = [x for x in v if x is not None]
        res[a] = dict(n=len(v), acc=round(sum(v) / max(1, len(v)), 3), ci95=wilson(sum(v), len(v)))
    return res


def paired(ok_gen, ok_real):
    """Separate the model's errors from the checker's. Only clips where the checker reads the REAL clip correctly count:
    there, a wrong reading of the generated clip is (very likely) the model's fault.
    model_error = wrong on generated / clips the checker can read; checker_blind = share of clips it can't read."""
    res = {}
    for a in ok_gen:
        pairs = [(g, r) for g, r in zip(ok_gen[a], ok_real[a]) if r is not None]
        readable = [g for g, r in pairs if r]
        wrong = len(readable) - sum(readable)
        res[a] = dict(readable=len(readable), model_errors=wrong,
                      model_error_rate=round(wrong / max(1, len(readable)), 3), ci95=wilson(wrong, len(readable)),
                      checker_blind=round(1 - len(readable) / max(1, len(pairs)), 3))
    return res


def mistakes_sheet(videos, truth, pred, bad, path, cols=6, tile=160):
    bad = bad[:24]
    if not bad:
        return
    rows = math.ceil(len(bad) / cols)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 48)), "white")
    d = ImageDraw.Draw(sheet)
    for n, i in enumerate(bad):
        x, y = (n % cols) * tile, (n // cols) * (tile + 48)
        sheet.paste(Image.fromarray(videos[i, 8].numpy()).resize((tile, tile)), (x, y))
        wrong = [f"{a}: {pred[a][i]} (want {truth[i][a]})" for a in ATTRS if truth[i][a] is not None and pred[a][i] != truth[i][a]]
        for k, line in enumerate(wrong[:4]):
            d.text((x + 2, y + tile + 1 + 11 * k), line[:30], fill=(170, 30, 30))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--meta", required=True, help="meta.jsonl: one clip metadata json per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_gen", type=int, default=256, help="stickers generated per split")
    ap.add_argument("--n_pool", type=int, default=3000, help="real training clips the checkers compare against")
    ap.add_argument("--skip_loss", action="store_true")
    ap.add_argument("--effect_v4_pool", default=None,
                    help="effect v4 reference file; built from real training clips if missing (copy it between runs so "
                         "both are judged by the same checker). Without it, effects are read by the v3 CLIP checker")
    ap.add_argument("--effect_v4_per_cell", type=int, default=40, help="real clips per subject + effect in that pool")
    ap.add_argument("--save_videos", action="store_true", help="save every decoded clip as uint8 (≈ 214 MB per 256)")
    ap.add_argument("--guidance", default="cfg:5",
                    help="comma list of ways to sample, compared on the same captions and seeds: cfg:W, cfgzero:W, auto:W:STEP "
                         "(autoguidance with the EMA snapshot from STEP as the weak model)")
    a = ap.parse_args()
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = {m["id"]: m for m in map(json.loads, open(a.meta))}
    splits = Path(a.splits)
    snaps = sorted((Path(a.run) / "snapshots").glob("ema_step*.pt"))
    report = dict(snapshots=[p.name for p in snaps])
    t0 = time.time()

    data = {s: Cache(a.cache, splits / f"{s}.txt", dev) for s in ("test_seen", "test_unseen", "val")}
    print("loaded caches", {s: len(d) for s, d in data.items()}, f"{time.time() - t0:.0f}s", flush=True)

    # ---- Part A
    if not a.skip_loss:
        curve = []
        for p in snaps:
            model, step = load_ema(p, dev)
            row = dict(step=step)
            for s, d in data.items():
                loss, t = per_clip_loss(model, d, dev)
                row[s] = round(loss.mean().item(), 5)
                if p == snaps[-1]:                                      # final snapshot: break the loss down
                    cats = defaultdict(list)
                    for c, l in zip(d.ids, loss.tolist()):
                        cats[meta[c]["category"]].append(l)
                    row[f"{s}_by_category"] = {k: round(float(np.mean(v)), 5) for k, v in sorted(cats.items())}
                    bins = defaultdict(list)
                    for tt, l in zip(t.tolist(), loss.tolist()):
                        bins["low t<0.35" if tt < 0.35 else "mid" if tt < 0.65 else "high t>=0.65"].append(l)
                    row[f"{s}_by_noise"] = {k: round(float(np.mean(v)), 5) for k, v in bins.items()}
            row["gap_unseen_minus_seen"] = round(row["test_unseen"] - row["test_seen"], 5)
            curve.append(row)
            print(json.dumps({k: v for k, v in row.items() if not isinstance(v, dict)}), flush=True)
            del model; torch.cuda.empty_cache()
        report["loss"] = curve
        (out / "report.json").write_text(json.dumps(report, indent=1))

    # ---- Part B
    from diffusers import AutoencoderKLWan
    from transformers import CLIPModel, CLIPProcessor
    from safetensors import safe_open

    model, step = load_ema(snaps[-1], dev)
    with safe_open(str(Path(a.cache) / "text" / "text.safetensors"), "pt") as f:
        empty = (f.get_slice("emb")[0:1].float().to(dev), f.get_slice("mask")[0:1].to(dev))
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae",
                                           torch_dtype=torch.float32).cuda().eval()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # the checkers' reference: real training clips, decoded the same way
    # building it takes ~20 min (mostly CPU-side CLIP preprocessing), so it is saved and reused between runs
    pool_file = Path(a.run) / f"eval_pool_{a.n_pool}.pt"
    if pool_file.exists():
        saved = torch.load(pool_file)
        pool_ids, pool = saved["ids"], saved["features"]
    else:
        train = Cache(a.cache, splits / "train.txt", "cpu")
        pool_ids = pick_spread(train.ids, meta, a.n_pool, seed=1)
        pos = {c: i for i, c in enumerate(train.ids)}
        pool_vid = decode(vae, train.normalize(train.latents[[pos[c] for c in pool_ids]]), train)
        pool = features(pool_vid, clip_model, proc)
        torch.save(dict(ids=pool_ids, features=pool), pool_file)
        del pool_vid, train
    pool_lab = {k: [labels(meta[c])[k] for c in pool_ids] for k in ATTRS}
    bg_palette = {}
    for m in meta.values():
        h = m["background"].lstrip("#")
        bg_palette[m["background_name"]] = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    print("pool ready", len(pool_ids), f"{time.time() - t0:.0f}s", flush=True)
    effect_v4 = None
    if a.effect_v4_pool:
        ef_file = Path(a.effect_v4_pool)
        if not ef_file.exists():
            train = Cache(a.cache, splits / "train.txt", "cpu")
            cells = defaultdict(list)
            for c in train.ids:
                cells[(f"{meta[c]['category']}/{meta[c]['subcategory']}", meta[c]["effect"])].append(c)
            rng = random.Random(3)
            ef_ids = []
            for k in sorted(cells):
                v = sorted(cells[k]); rng.shuffle(v); ef_ids += v[:a.effect_v4_per_cell]
            pos = {c: i for i, c in enumerate(train.ids)}
            ef_feats = []
            for i in range(0, len(ef_ids), 256):                       # decode in chunks: 256 clips ≈ 214 MB
                vid = decode(vae, train.normalize(train.latents[[pos[c] for c in ef_ids[i:i + 256]]]), train)
                bgs = torch.stack([border_colour(v.cuda()).cpu() for v in vid])
                ef_feats.append(effect_features(vid, bgs))
                del vid
            torch.save(dict(ids=ef_ids, features=np.concatenate(ef_feats)), ef_file)
            del train
        saved = torch.load(ef_file, weights_only=False)
        by = defaultdict(lambda: ([], []))
        for c, x in zip(saved["ids"], saved["features"]):
            s = f"{meta[c]['category']}/{meta[c]['subcategory']}"
            by[s][0].append(x); by[s][1].append(meta[c]["effect"])
        effect_v4 = effect_check.Checker({s: (np.stack(x), l) for s, (x, l) in by.items()})
        print("effect v4 pool ready", len(saved["ids"]), f"{time.time() - t0:.0f}s", flush=True)
    allowed = defaultdict(set)                                          # motions the generator gives each subject
    for m in meta.values():
        allowed[f"{m['category']}/{m['subcategory']}"].add(m["motion"])

    weak = {}
    for g in a.guidance.split(","):
        if g.startswith("auto:"):
            st = int(g.split(":")[2])
            if st not in weak:
                weak[st] = load_ema(Path(a.run) / "snapshots" / f"ema_step{st:06d}.pt", dev)[0]
    report["generation"] = dict(step=step, n_per_split=a.n_gen, steps=50, shift=5.0, n_pool=len(pool_ids),
                                guidance=a.guidance.split(","))
    for s in ("test_seen", "test_unseen"):
        d = data[s]
        ids = pick_spread(d.ids, meta, a.n_gen, seed=2)
        pos = {c: i for i, c in enumerate(d.ids)}
        idx = [pos[c] for c in ids]
        rows = torch.tensor([d.rows[i][0] for i in idx], device=dev)
        truth = [labels(meta[c]) for c in ids]
        real_vid = decode(vae, d.normalize(d.latents[idx].to(dev)), d)
        real_pred = predict(features(real_vid, clip_model, proc), pool, pool_lab, bg_palette, real_vid, allowed, effect_v4)
        if a.save_videos:
            torch.save(dict(ids=ids, videos=real_vid), out / f"videos_{s}_real.pt")
        real_ok = correct(real_pred, truth)
        report[s] = dict(real_ceiling=score(real_ok), clips=ids, real=real_pred)
        blind = [i for i in range(len(ids)) if not real_ok["all"][i]]
        mistakes_sheet(real_vid, truth, real_pred, blind, out / f"checker_errors_on_real_{s}.png")
        del real_vid
        for gname in a.guidance.split(","):
            parts = gname.split(":")
            kw = dict(guide=parts[0], cfg=float(parts[1]))
            if parts[0] == "auto":
                kw["bad_model"] = weak[int(parts[2])]
            gen = []
            for i in range(0, len(idx), 64):
                r = rows[i:i + 64]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = sample(model, d.emb[r].float(), d.mask[r], *empty, seed=1000 + i, **kw)
                gen.append(z.float().cpu())
            gen_vid = decode(vae, torch.cat(gen), d)
            gen_pred = predict(features(gen_vid, clip_model, proc), pool, pool_lab, bg_palette, gen_vid, allowed, effect_v4)
            gen_ok = correct(gen_pred, truth)
            res = dict(generated=score(gen_ok), paired=paired(gen_ok, real_ok), pred=gen_pred)
            model_bad = [i for i in range(len(ids)) if real_ok["all"][i] and not gen_ok["all"][i]]
            mistakes_sheet(gen_vid, truth, gen_pred, model_bad, out / f"model_errors_{s}_{gname.replace(':', '-')}.png")
            conf = Counter((t["motion"], p) for t, p in zip(truth, gen_pred["motion"]) if t["motion"] not in (None, p))
            res["motion_confusions"] = [f"{t} -> {p}: {n}" for (t, p), n in conf.most_common(8)]
            report[s][gname] = res
            if a.save_videos:                                           # every clip, so later checkers need no GPU
                torch.save(dict(ids=ids, videos=gen_vid), out / f"videos_{s}_{gname.replace(':', '-')}.pt")
            else:
                torch.save(dict(ids=ids, generated=gen_vid[:32].clone()), out / f"samples_{s}_{gname.replace(':', '-')}.pt")
            print(s, gname, "acc", json.dumps({k: v["acc"] for k, v in res["generated"].items()}), flush=True)
            print(s, gname, "model_error", json.dumps({k: v["model_error_rate"] for k, v in res["paired"].items()}),
                  f"ram {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
            del gen, gen_vid, gen_pred                                  # one condition's videos at a time (~1 GB each)
            gc.collect()
            (out / "report.json").write_text(json.dumps(report, indent=1))
        print(s, "real_ceiling", json.dumps({k: v["acc"] for k, v in report[s]["real_ceiling"].items()}), flush=True)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
