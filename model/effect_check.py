"""Effect checker v4: read which effect a sticker has from its particles, not from CLIP.  (OUR code.)

Why: the v3 checker read effects from CLIP embeddings. It was 96–97% right on real clips but, by eye, called visible
rocket exhaust, sweat drops, tears and Zzz "none" on generated clips (runs/r2/RESULTS.md, visual audit). CLIP sees the
whole picture at 224 px; a 6-pixel drop barely moves it, and generated particles are softer than real ones.

What the generator draws (generator/bhaav_gen.py, effect_*): every effect is small particles laid on top of the subject,
in fixed colours, mostly *outside* the subject's body:
  tears, tears-joy, sweat   blue drops (90,170,255) / (120,200,255), falling or arcing away from the face
  hearts                    pink-red (255,70,110), floating up from the top edge
  sparkles                  yellow 4-point stars (255,230,90) at the corners of the art box, twinkling
  steam, exhaust            grey or white puffs (above the head / down-left from the tail)
  zzz                       blue-grey or white Z strokes, up and right
  thought                   white circles with a grey outline, above and right
  confetti                  14 bits in 6 colours bursting out once per second
  embers                    tiny orange / yellow dots rising from the flame

So per frame: the subject = the largest blob that differs from the background (grown by a few pixels). Everything else
that differs from the background is a particle. Over all 17 frames we measure how much particle area there is in each
colour family, where it sits relative to the body, how many pieces, and how much it changes over time.

Some subjects carry particle-like art of their own (the party popper's streamers, the sleepy face's painted Zzz), so the
features are not read with fixed thresholds. Each clip is compared with real training clips of the same subject
(k nearest neighbours, features standardised within the subject), and only that subject's possible effects are allowed.
"""
import numpy as np
from scipy import ndimage

BODY_THR, PART_THR, GROW = 45.0, 28.0, 5          # colour distance from background (0–441), body growth in pixels
FAMILIES = ["neutral", "blue", "pink", "orange", "yellow", "green", "purple"]
NAMES = ([f"area_{f}" for f in FAMILIES] + ["area_std", "pieces", "dx", "dy", "spread"]
         + [f"on_body_{f}" for f in FAMILIES] + ["on_body_change"])


def _hsv(rgb):
    """rgb float 0..1, (..., 3) -> hue in degrees, saturation, value."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn + 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6), mx


def _family(rgb):
    h, s, v = _hsv(rgb)
    fam = np.full(h.shape, 6)                        # purple by default, then overwrite
    fam[(h >= 70) & (h < 185)] = 5                   # green / teal
    fam[(h >= 40) & (h < 70)] = 4                    # yellow
    fam[(h >= 8) & (h < 40)] = 3                     # orange
    fam[(h >= 320) | (h < 8)] = 2                    # pink-red
    fam[(h >= 185) & (h < 235)] = 1                  # blue
    fam[s < 0.18] = 0                                # grey / white, whatever the hue
    return fam


def features(video, bg):
    """video: uint8 (T, H, W, 3); bg: [r, g, b]. Returns a float vector named by NAMES."""
    v = video.astype(np.float32)
    dist = np.linalg.norm(v - np.asarray(bg, np.float32), axis=-1)          # (T, H, W)
    areas, pieces, dxs, dys, spreads, on_body, hists = [], [], [], [], [], [], []
    for t in range(len(v)):
        lab, n = ndimage.label(dist[t] > BODY_THR)
        if n == 0:
            areas.append(np.zeros(len(FAMILIES))); on_body.append(np.zeros(len(FAMILIES)))
            hists.append(np.zeros(len(FAMILIES))); pieces.append(0); continue
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
        body = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
        ys, xs = np.nonzero(body)
        cy, cx, size = ys.mean(), xs.mean(), max(8.0, np.sqrt(body.sum()))
        part = (dist[t] > PART_THR) & ~ndimage.binary_dilation(body, iterations=GROW)
        part = ndimage.binary_opening(part)                                 # drop single-pixel speckle
        # effects that never leave the face (a sweat drop sliding down the forehead) only show up as colour on the body;
        # the subject's own colours are the baseline, set by the same subject's real clips in the Checker
        body_fam = np.bincount(_family(v[t][body] / 255.0), minlength=len(FAMILIES)).astype(np.float32)
        on_body.append(body_fam)
        hists.append(body_fam / max(1.0, body_fam.sum()))
        fam = _family(v[t][part] / 255.0)
        areas.append(np.bincount(fam, minlength=len(FAMILIES)).astype(np.float32))
        pieces.append(ndimage.label(part)[1])
        if part.any():
            py, px = np.nonzero(part)
            dxs.append((px.mean() - cx) / size); dys.append((py.mean() - cy) / size)
            spreads.append(np.hypot(px - px.mean(), py - py.mean()).mean() / size)
    areas, on_body, hists = np.stack(areas), np.stack(on_body), np.stack(hists)
    total = areas.sum(1)
    mean = lambda x: float(np.mean(x)) if len(x) else 0.0
    change = np.abs(np.diff(hists, axis=0)).sum(1).mean()        # how much the body's colour mix changes frame to frame
    return np.concatenate([np.log1p(areas.mean(0)), [np.log1p(total.std()), np.log1p(np.mean(pieces)),
                                                      mean(dxs), mean(dys), mean(spreads)],
                           np.log1p(on_body.mean(0)), [change * 10]]).astype(np.float32)


class Checker:
    """k-NN over real clips of the same subject. pool: {subject: (features (N, F), effect labels [N])}."""

    def __init__(self, pool, k=7):
        self.k, self.pool = k, {}
        for subj, (f, lab) in pool.items():
            mu, sd = f.mean(0), f.std(0) + 0.25     # floor on sd: a feature that never varies must not dominate
            self.pool[subj] = ((f - mu) / sd, np.asarray(lab), mu, sd)

    def predict(self, feat, subject):
        f, lab, mu, sd = self.pool[subject]
        d = np.linalg.norm(f - (feat - mu) / sd, axis=1)
        near = lab[np.argsort(d)[:self.k]]
        vals, counts = np.unique(near, return_counts=True)
        return str(vals[np.argmax(counts)])
