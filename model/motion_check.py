"""Motion checker v3: compare a clip's movement with the generator's own motion formulas.  (OUR code.)

Why: the v2 checker found the nearest real clips by movement numbers and copied their label. It mixed up pulse, float and
still, because those motions are small (pulse = size +-8%, float = up/down +-10 px), and the neighbours didn't know what
each motion is *supposed* to look like. But we wrote the generator, so we know the exact formulas
(generator/bhaav_gen.py, motion_*). This checker asks, for each motion: "if the clip moved like this formula, at the best
speed and starting point, how well would that explain what we measure?" The best explanation wins.

What we measure per frame, from the object's box (10%-90% extents of the foreground, so small particles barely matter):
  x, y   box centre, in pixels, minus its average  (shake moves x; bounce and float move y)
  s      box width / average width, minus 1        (pulse, heartbeat, flicker, and bounce's squash change it)
  a      total turn in degrees since frame 0, by matching each frame's picture rotated against the next
                                                    (wobble +-12, float +-3, spin, flicker)
How we score each motion: squared distance between measured and predicted curves, each channel divided by its typical
measurement noise (sigma), so a pixel of x and a percent of size count fairly. That's a Gaussian likelihood: the
motion with the smallest score is the most likely one.
"""
import math

import numpy as np

FPS, FRAMES = 8, 17


# ---- the generator's formulas, copied as plain functions of the cycle position p (0..1) ----
# each returns (dx px, dy px, sx, sy, angle deg); keep in sync with generator/bhaav_gen.py

def _bounce(p, amp=28):
    h = abs(math.sin(math.pi * p)); sq = (1 - h) ** 3
    return 0.0, amp / 2 - amp * h, 1 + 0.06 * sq, 1 - 0.08 * sq, 0.0


def _heartbeat(p):
    b = max(math.exp(-((p - 0.1) / 0.07) ** 2), 0.7 * math.exp(-((p - 0.35) / 0.07) ** 2))
    return 0.0, 0.0, 1 + 0.15 * b, 1 + 0.15 * b, 0.0


def _flicker(p):
    k = math.sin(2 * math.pi * p)
    return 0.0, 0.0, 1 - 0.05 * k, 1 + 0.07 * k, 3 * math.sin(4 * math.pi * p)


FORMULAS = {
    "bounce": (_bounce, [0.5, 1.0]),
    "wobble": (lambda p: (0.0, 0.0, 1.0, 1.0, 12 * math.sin(2 * math.pi * p)), [0.5, 1.0]),
    "pulse": (lambda p: (0.0, 0.0, 1 + 0.08 * math.sin(2 * math.pi * p), 1 + 0.08 * math.sin(2 * math.pi * p), 0.0), [0.5, 1.0]),
    "float": (lambda p: (0.0, 10 * math.sin(2 * math.pi * p), 1.0, 1.0, 3 * math.sin(2 * math.pi * p + 1)), [0.5]),
    "shake": (lambda p: (9 * math.sin(2 * math.pi * p), 0.0, 1.0, 1.0, 0.0), [1.0, 2.0]),
    "still": (lambda p: (0.0, 0.0, 1.0, 1.0, 0.0), [0.5]),
    "spin": (lambda p: (0.0, 0.0, 1.0, 1.0, -360 * p), [0.5]),
    "heartbeat": (_heartbeat, [1.0]),
    "flicker": (_flicker, [1.0, 2.0]),
}
PHASES = np.linspace(0, 1, 24, endpoint=False)


def _templates():
    """Every motion x speed x starting phase -> the 17-frame curves we would measure (x, y, s, a), each minus its mean."""
    out = {}
    t = np.arange(FRAMES) / FPS
    for name, (fn, speeds) in FORMULAS.items():
        curves = []
        for hz in speeds:
            for ph in PHASES:
                v = np.array([fn((hz * tt + ph) % 1.0) for tt in t])          # [17, 5]
                x, y, sx, sy, a = v.T
                y = y + 85 * (1 - sy)                                          # squash is anchored at the bottom edge
                c = np.stack([x, y, sx - 1, a], 1)
                curves.append(c - c.mean(0))
        if name == "spin":                                                    # measured tilt may run either way
            curves += [c * np.array([1, 1, 1, -1]) for c in curves]
        out[name] = np.stack(curves)                                           # [n, 17, 4]
    return out


TEMPLATES = _templates()


def _extent(profile, lo=0.1, hi=0.9):
    cdf = np.cumsum(profile, -1) / np.maximum(profile.sum(-1, keepdims=True), 1)
    return (cdf < lo).sum(-1).astype(float), (cdf < hi).sum(-1).astype(float)


def _rotation_steps(video, cx, cy, crop=48, max_deg=30.0, step=1.5):
    """How many degrees the object turns between each pair of frames: rotate frame t's picture (around its own centre)
    by each candidate angle and keep the angle that best matches frame t+1. Works on the picture, not the outline, so a
    round face still shows its turn through the eyes and mouth, and a 5-pointed star can't hide a spin.
    Runs at half resolution for speed."""
    g = video.astype(np.float32).mean(-1)[:, ::2, ::2]                       # [17, 128, 128]
    T, H, W = g.shape
    r = np.arange(-crop, crop, dtype=np.float32)
    yy, xx = np.meshgrid(r, r, indexing="ij")
    keep = (xx ** 2 + yy ** 2) < crop ** 2                                    # round window: corners don't bias the angle
    def sample(f, ox, oy, deg):
        a = math.radians(deg)
        xs = ox + xx * math.cos(a) - yy * math.sin(a)
        ys = oy + xx * math.sin(a) + yy * math.cos(a)
        return f[np.clip(ys.round().astype(int), 0, H - 1), np.clip(xs.round().astype(int), 0, W - 1)]
    angles = np.arange(-max_deg, max_deg + 1e-6, step)
    out = []
    for k in range(T - 1):
        ref = sample(g[k + 1], cx[k + 1] / 2, cy[k + 1] / 2, 0.0)[keep]
        err = [np.abs(sample(g[k], cx[k] / 2, cy[k] / 2, d)[keep] - ref).mean() for d in angles]
        out.append(angles[int(np.argmin(err))])
    return np.array(out)


def measure(video, bg):
    """video uint8 [17, H, W, 3], bg [3] -> curves [17, 4] (x, y, s, a), each minus its mean."""
    c = video.astype(np.float32)
    fg = (np.linalg.norm(c - np.asarray(bg, np.float32), axis=-1) > 40).astype(np.float32)   # [17, H, W]
    x0, x1 = _extent(fg.sum(1))
    y0, y1 = _extent(fg.sum(2))
    w = np.maximum(x1 - x0, 1)
    H, W = fg.shape[1:]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    area = np.maximum(fg.sum((1, 2)), 1)
    cx, cy = (fg * xs).sum((1, 2)) / area, (fg * ys).sum((1, 2)) / area
    dx, dy = xs - cx[:, None, None], ys - cy[:, None, None]
    mxx, myy, mxy = [(fg * q).sum((1, 2)) / area for q in (dx * dx, dy * dy, dx * dy)]
    ang = np.concatenate([[0.0], np.cumsum(_rotation_steps(video, cx, cy))])   # total turn since frame 0
    gyr = np.sqrt(mxx + myy)                                                  # average distance of the shape from its centre
    m = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, w / w.mean() - 1, ang], 1)
    m_gyr = gyr / gyr.mean() - 1                                              # size from spread (used for UI, see classify)
    m = m - m.mean(0)
    return m, m_gyr


SIGMA = np.array([1.5, 1.5, 0.012, 2.0])            # typical measurement noise per channel (px, px, fraction, deg)


def classify(video, bg, allowed=None, ui=False, return_scores=False):
    """allowed: the motions the generator can give this subject (read from the data, e.g. UI icons only get still / float /
    pulse on top of their own animation). Without it, a loading spinner turning on its own looks like "spin".
    ui: UI icons animate themselves (spinner arc, toggle knob), so their tilt says nothing about the added motion."""
    m, m_gyr = measure(video, bg)
    sigma = SIGMA.copy()
    if ui:
        # UI icons animate themselves (spinner arc turning, progress filling), which moves the box edges and turns the
        # picture. Use size-from-spread instead of box width (a turning arc keeps its spread) and don't trust the turn.
        m[:, 2] = m_gyr - m_gyr.mean()
        sigma[3] = 1e3
    scores = {}
    for name, tpl in TEMPLATES.items():
        if allowed is not None and name not in allowed:
            continue
        d = ((tpl - m[None]) / sigma) ** 2           # [n, 17, 4]
        scores[name] = float(d.sum((1, 2)).min())
    best = min(scores, key=scores.get)
    return (best, scores) if return_scores else best
