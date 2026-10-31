"""Step 1 of adding sound: can a frozen audio compressor rebuild our sounds?  (OUR code.)

Same question as vae_roundtrip.py asked of the Wan2.2 VAE: if the compressor smears our sounds, no generator trained
on its latents can make them. Our sounds are short clicks, pops and plinks whose *timing* is the point, so the main
test is whether each sound still starts at the same moment after the round trip.

For each of 200 clips (spread over every effect, motion and UI element): make the 2 s sound loop from the clip's spec
(generator/bhaav_audio.py), repeat it 3 times so the compressor sees it as a loop, encode -> decode, keep the middle
2 s, and compare with the original.

Pass marks, fixed before the first run:
  - onsets kept:  >= 97% of the sound onsets found in the original are still found after the round trip
  - onset shift:  95th percentile |shift| <= 10 ms   (our sync pass mark for the model is 50 ms)
  - by ear: the listening pairs in data/audio_gate/<codec>/ sound like the same sounds
Spectral distance and SNR are reported for comparing codecs; they have no pass mark.

Codecs:
  dac44   Descript Audio Codec 44.1 kHz (MIT), round trip through its 9 codebooks
  dac16   Descript Audio Codec 16 kHz (MIT), 12 codebooks
  identity, smear   controls: no change (must pass) and a 25 ms blur (must fail)
  sao     Stable Audio Open 1.0 VAE (Stability AI Community License; gated: accept it on Hugging Face first)

Run:  HF_HUB_OFFLINE=0 .venv/bin/python gpu/audio_vae_gate.py --codec dac44 --n 200
"""
import argparse
import json
import random
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generator"))
import bhaav_audio as A  # noqa: E402
import bhaav_gen as G  # noqa: E402

SR = A.SR                                            # 32 kHz: what bhaav_audio makes


# ---------- codecs: each takes [B, T] float at SR and returns the rebuilt [B, T] plus its latent shape ----------

class Codec:
    def __init__(self, name, dev):
        self.name, self.dev = name, dev
        if name in ("identity", "smear"):           # controls: the gate must pass identity and fail smear
            self.sr = SR
        elif name.startswith("dac"):
            import dac
            self.m = dac.DAC.load(dac.utils.download(model_type={"dac44": "44khz", "dac16": "16khz"}[name])).to(dev).eval()
            self.sr = self.m.sample_rate
        elif name == "sao":
            from diffusers import AutoencoderOobleck
            self.m = AutoencoderOobleck.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae").to(dev).eval()
            self.sr = self.m.config.sampling_rate
        else:
            raise ValueError(name)

    @torch.no_grad()
    def roundtrip(self, x):
        if self.name == "identity":
            return x.copy(), ("none",)
        if self.name == "smear":                       # 25 ms blur in time: what a bad compressor does to clicks
            k = np.hanning(int(0.025 * SR)); k /= k.sum()
            return np.stack([np.convolve(r, k, mode="same") for r in x]), ("none",)
        y = resample_poly(x, self.sr // np.gcd(self.sr, SR), SR // np.gcd(self.sr, SR), axis=-1)
        t = torch.from_numpy(y.astype(np.float32)).to(self.dev)[:, None]          # [B, 1, T]
        if self.name.startswith("dac"):
            n = t.shape[-1]
            z, codes, *_ = self.m.encode(self.m.preprocess(t, self.sr))
            out = self.m.decode(z)[..., :n]
            shape = tuple(z.shape[1:])
        else:                                                                       # VAE expects stereo
            z = self.m.encode(t.expand(-1, 2, -1)).latent_dist.mean
            out = self.m.decode(z).sample.mean(1, keepdim=True)[..., :t.shape[-1]]
            shape = tuple(z.shape[1:])
        out = out[:, 0].float().cpu().numpy()
        back = resample_poly(out, SR // np.gcd(self.sr, SR), self.sr // np.gcd(self.sr, SR), axis=-1)
        return back[:, :x.shape[-1]], shape


# ---------- measures ----------

H, W = 64, 512                                       # 2 ms hop, 16 ms window


def flux(x):
    """Spectral flux of a 2 s loop (computed on 3 copies, so the loop edges are handled): [frames of one loop]."""
    x3 = np.tile(x, 3)
    fr = np.lib.stride_tricks.sliding_window_view(x3, W)[::H] * np.hanning(W)
    S = np.log1p(100 * np.abs(np.fft.rfft(fr, axis=1)))
    f = np.concatenate([[0], np.maximum(0, np.diff(S, axis=0)).sum(1)])
    return f


def onsets(f, events):
    """For each scheduled event: the onset time (ms after schedule) found near it, or None."""
    thr = np.median(f) + 0.75 * f.std()
    found = []
    for t, *_ in events:
        c = int((A.N + t * SR - W / 2) / H)          # the middle copy
        win = f[c - 3:c + 12]
        k = int(np.argmax(win))
        found.append((k - 3) * H / SR * 1000 if win[k] > thr else None)
    return found


def log_spec_dist(a, b):
    """Mean |log magnitude difference| over two STFT sizes (dB-like; 0 = identical spectra)."""
    d = []
    for w in (256, 1024):
        fa = np.lib.stride_tricks.sliding_window_view(a, w)[::w // 4] * np.hanning(w)
        fb = np.lib.stride_tricks.sliding_window_view(b, w)[::w // 4] * np.hanning(w)
        ma, mb = np.abs(np.fft.rfft(fa)), np.abs(np.fft.rfft(fb))
        d.append(np.mean(np.abs(20 * np.log10(ma + 1e-4) - 20 * np.log10(mb + 1e-4))))
    return float(np.mean(d))


def snr(a, b):
    return float(10 * np.log10((a ** 2).sum() / (((a - b) ** 2).sum() + 1e-12)))


# ---------- clips ----------

def pick(n, seed=0):
    """Round-robin over every effect, motion and UI element, so rare sounds are tested too."""
    rng = random.Random(seed)
    ids = A.manifest()
    rng.shuffle(ids)
    groups = defaultdict(list)
    for _, p in ids[:6000]:
        c = json.loads(p.read_text())
        if c.get("generator_version", "0") < "0.3":
            continue
        groups[("ui", c["subcategory"]) if c["category"] == "ui" else ("fx", c["effect"])].append(c)
        groups[("motion", c["motion"])].append(c)
    out, seen = [], set()
    while len(out) < n:
        for g in sorted(groups):
            while groups[g] and groups[g][-1]["id"] in seen:
                groups[g].pop()
            if groups[g] and len(out) < n:
                c = groups[g].pop()
                seen.add(c["id"])
                out.append(c)
    return out


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(np.tile(x, 2), -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", required=True, choices=["dac44", "dac16", "sao", "identity", "smear"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--listen", type=int, default=12)
    a = ap.parse_args()

    out = G.DATA / "audio_gate" / a.codec
    out.mkdir(parents=True, exist_ok=True)
    codec = Codec(a.codec, a.device)
    clips = pick(a.n)
    rows, t0, shape = [], time.time(), None
    for i in range(0, len(clips), a.batch):
        batch = clips[i:i + a.batch]
        xs = [A.render_audio(c) for c in batch]
        loops = np.stack([np.tile(x, 3) for x, _ in xs])
        rec, shape = codec.roundtrip(loops)
        for c, (x, ev), r in zip(batch, xs, rec):
            r = r[A.N:2 * A.N]                         # the middle loop
            o_on, r_on = onsets(flux(x), ev), onsets(flux(r), ev)
            shifts = [b - a_ for a_, b in zip(o_on, r_on) if a_ is not None and b is not None]
            rows.append({"id": c["id"], "effect": c["effect"], "motion": c["motion"],
                         "ui": c["subcategory"] if c["category"] == "ui" else None,
                         "onsets_orig": sum(v is not None for v in o_on), "onsets_kept": len(shifts),
                         "shifts_ms": shifts, "lsd": round(log_spec_dist(x, r), 2), "snr": round(snr(x, r), 1)})
            if a.listen and (len(rows) - 1) % max(1, a.n // a.listen) == 0:    # spread the listening pairs over the sample
                write_wav(out / f"{c['id']}_orig.wav", x)
                write_wav(out / f"{c['id']}_{a.codec}.wav", r)
    dt = time.time() - t0

    orig = sum(r["onsets_orig"] for r in rows)
    kept = sum(r["onsets_kept"] for r in rows)
    sh = np.abs(np.concatenate([r["shifts_ms"] for r in rows if r["shifts_ms"]]))
    by = defaultdict(list)
    for r in rows:
        by[r["ui"] or r["effect"]].append(r["lsd"])
    worst = sorted(((np.mean(v), k) for k, v in by.items()), reverse=True)[:5]
    summary = {
        "codec": a.codec, "clips": len(rows), "latent_per_6s_input": shape,
        "onsets_kept": f"{kept}/{orig} = {kept / orig:.1%}", "shift_ms_median": float(np.median(sh)),
        "shift_ms_p95": float(np.percentile(sh, 95)),
        "lsd_mean": round(float(np.mean([r["lsd"] for r in rows])), 2),
        "snr_median_db": float(np.median([r["snr"] for r in rows])),
        "worst_lsd_groups": [(k, round(float(v), 2)) for v, k in worst],
        "clips_per_s": round(len(rows) / dt, 1), "device": a.device,
        "pass": bool(kept / orig >= 0.97 and np.percentile(sh, 95) <= 10),
    }
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
