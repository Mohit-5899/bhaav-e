"""Bhaav-E sound generator (our code).

Adds a 2-second looping soundtrack to a clip that bhaav_gen.py already made, using only the clip's <id>.json.
No recordings and no sample library: every sound is synthesised here, so there is nothing to license.

How the timing is recovered: make_clip() seeds random.Random(cfg["seed"]) and its first two draws are the effect
phases. Every effect emits on a fixed schedule from those phases (ages(): a particle is born whenever
t = phase (mod every)), and every motion / UI animation is a function of p = (speed * t + phase) mod 1.
So the time of every visual event, and therefore every sound, can be computed from the spec without re-rendering.

Each clip's sound is written into a circular 2.0 s buffer, so the soundtrack loops with no seam, like the video.
Alongside the audio we save the event list (time, kind): the ground truth a sync metric can be scored against.

Usage:
  sample:  .venv/bin/python generator/bhaav_audio.py sample --n 20      # writes data/audio_sample/
  check:   .venv/bin/python generator/bhaav_audio.py check --n 200     # do specs + seeds still reproduce the videos?
"""

import argparse
import json
import math
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bhaav_gen as G  # noqa: E402

SR = 32000                     # samples per second
N = int(G.DURATION * SR)       # 64,000 samples = exactly one 2.0 s loop
OUT = G.DATA / "audio_sample"


# ---------- when things happen (seconds in 0..2) ----------

def births(offset, every):
    """Times a particle stream emits: t = offset (mod every)."""
    t0 = offset % every
    return [t0 + k * every for k in range(int(round(G.DURATION / every)))]


def at_phase(p_event, speed, phase):
    """Times where (speed * t + phase) mod 1 == p_event."""
    t0 = ((p_event - phase) % 1.0) / speed
    return [t0 + k / speed for k in range(int(round(G.DURATION * speed)))]


def effect_events(cfg, ph):
    """Mirrors the EFFECTS functions in bhaav_gen.py: (time, sound, variant)."""
    e, ev = cfg["effect"], []
    if e == "tears":
        ev = [(t, "drip", side) for side in (0, 1) for t in births(ph[side], 0.5)]
    elif e == "tears-joy":
        ev = [(t, "drip_small", side) for side in (0, 1) for t in births(ph[side], 0.25)]
    elif e == "hearts":
        ev = [(t, "pop", i) for i in range(3) for t in births(ph[i % 2] + i * 0.33, 1.0)]
    elif e == "sparkles":                            # twinkle = sin(2 pi (t + ph + i/4)) peaks at 0.25
        ev = [(t, "chime", i) for i in range(4) for t in births(0.25 - ph[i % 2] - i * 0.25, 1.0)]
    elif e == "sweat":
        ev = [(t, "slide", 0) for t in births(ph[0], 1.0)]
    elif e == "steam":
        ev = [(t, "hiss", side) for side in (0, 1) for t in births(ph[side], 0.25)]
    elif e == "zzz":
        ev = [(t, "zz", 0) for t in births(ph[0], 0.5)]
    elif e == "thought":                             # bubble i appears at p = (t + ph) mod 1 = 0.2 i
        ev = [(t, "bubble", i) for i in range(3) for t in births(0.2 * i - ph[0], 1.0)]
    elif e == "confetti":
        ev = [(t, "burst", 0) for t in births(ph[0], 1.0)]
    elif e == "embers":
        ev = [(t, "crackle", k) for k in range(4) for t in births(ph[0] + k * 0.13, 0.5)]
    elif e == "exhaust":
        ev = [(t, "puff", 0) for t in births(ph[0], 0.25)]
    return ev


MOTION_EVENTS = {                                    # p values where the motion "hits" something
    "bounce": [(0.0, "boing")],                      # |sin(pi p)| = 0: on the ground
    "heartbeat": [(0.1, "lub"), (0.35, "dub")],      # the two beat peaks in motion_heartbeat
    "shake": [(0.25, "rattle"), (0.75, "rattle")],   # the two ends of the shake
    "wobble": [(0.25, "creak"), (0.75, "creak")],    # the two ends of the tilt
    "pulse": [(0.25, "whoomp")],                     # biggest size
}
MOTION_BEDS = {"spin", "float", "flicker"}           # continuous sounds that follow p


def motion_events(cfg):
    return [(t, s, 0) for pe, s in MOTION_EVENTS.get(cfg["motion"], [])
            for t in at_phase(pe, cfg["speed_hz"], cfg.get("motion_phase", 0.0))]


UI_EVENTS = {                                        # p values read from the ui_* draw functions
    "spinner": [(0.0, "tick", 0)],
    "dots-loader": [(0.25 + 0.15 * k, "blip", k) for k in range(3)],
    "success": [(0.55, "ding", 0)],                  # the tick finishes drawing at p = 0.55
    "error": [(0.55, "buzz", 0)],
    "like": [(0.15, "tap", 0)],
    "bell": [(0.125, "bell", 0), (0.375, "bell", 1)],  # swing peaks before the swing dies at p = 0.6
    "download": [(0.6, "thunk", 0)],                 # the arrow lands in the tray
    "progress": [(0.0, "fill", 0), (0.7, "done", 0)],
    "toggle": [(0.35, "click", 1), (0.85, "click", 0)],  # knob arrives: on, then off
    "typing": [(0.25 + 0.17 * k, "key", k) for k in range(3)],
}


def ui_events(cfg):
    if cfg["category"] != "ui":
        return []
    return [(t, s, v) for pe, s, v in UI_EVENTS[cfg["subcategory"]]
            for t in at_phase(pe, cfg["ui_speed_hz"], cfg["ui_phase"])]


def events(cfg):
    ph = [random.Random(cfg["seed"]).uniform(0, 1) for _ in range(2)]   # same draws as make_clip
    return sorted((round(t % G.DURATION, 4), s, v) for t, s, v in effect_events(cfg, ph) + motion_events(cfg) + ui_events(cfg))


# ---------- the sounds ----------

def _t(dur):
    return np.arange(int(dur * SR)) / SR


def env(dur, attack, decay):
    t = _t(dur)
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / decay)


def glide(f0, f1, dur, decay, attack=0.002, vib=0.0, vib_hz=0.0):
    """A sine whose pitch slides from f0 to f1 (exponentially), with optional wobble."""
    t = _t(dur)
    f = f0 * (f1 / f0) ** (t / dur) * (1 + vib * np.sin(2 * np.pi * vib_hz * t) * np.exp(-t / 0.15))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(dur, attack, decay)


def band(x, lo, hi):
    """Keep only frequencies between lo and hi (circular FFT filter, so a 2 s bed stays loopable)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    y = np.fft.irfft(X, len(x))
    return y / (np.abs(y).max() + 1e-9)


def noise(rng, dur, lo, hi, attack, decay):
    return band(rng.standard_normal(int(dur * SR)), lo, hi) * env(dur, attack, decay)


def mix(*xs):
    """Add sounds of different lengths."""
    out = np.zeros(max(len(x) for x in xs))
    for x in xs:
        out[:len(x)] += x
    return out


def partials(f, ratios, dur, decay, amps=None):
    t = _t(dur)
    amps = amps or [1 / (k + 1) for k in range(len(ratios))]
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (decay / (1 + 0.5 * k)))
               for k, (r, a) in enumerate(zip(ratios, amps))) * np.minimum(1, t / 0.002)


PENTATONIC = [1568.0, 1760.0, 2093.0, 2349.0]      # G6 A6 C7 D7: sparkles never clash


def sound(kind, var, rng):
    """One event -> (samples, gain)."""
    if kind == "drip":                               # a water plink: pitch rises fast
        return glide(700 + 120 * var, 1700 + 200 * var, 0.09, 0.03), 0.55
    if kind == "drip_small":
        return glide(1100 + 150 * var, 2300, 0.05, 0.015), 0.3
    if kind == "pop":
        return mix(glide(280 + 60 * var, 900 + 120 * var, 0.07, 0.02), 0.3 * noise(rng, 0.07, 1500, 6000, 0.001, 0.004)), 0.5
    if kind == "chime":
        return partials(PENTATONIC[var], [1, 2.0, 3.01], 0.6, 0.18), 0.22
    if kind == "slide":                              # the sweat drop slides 0.6 s: a falling whistle
        x = glide(1900, 750, 0.6, 2.0, attack=0.04)
        return x * np.minimum(1, (0.6 - _t(0.6)) / 0.08), 0.25
    if kind == "hiss":
        return noise(rng, 0.3, 2500, 9000, 0.03, 0.09), 0.18
    if kind == "zz":                                 # buzzy low "zzz": harmonics of 95 Hz, swelling
        t = _t(0.35)
        x = sum(np.sin(2 * np.pi * 95 * k * t) / k for k in range(1, 14))
        return x / 3 * np.sin(np.pi * t / 0.35) ** 2, 0.3
    if kind == "bubble":
        return glide(500 * 1.25 ** var, 1100 * 1.25 ** var, 0.06, 0.02), 0.45
    if kind == "burst":                              # popper: low pop, then paper rustle
        return mix(glide(160, 520, 0.08, 0.03), 0.6 * noise(rng, 0.45, 1200, 7000, 0.003, 0.12)), 0.6
    if kind == "crackle":
        x = np.zeros(int(0.12 * SR))
        for _ in range(3):
            i = rng.integers(0, len(x) - 80)
            x[i:i + 80] += noise(rng, 80 / SR, 1500, 9000, 0.0002, 0.0012) * rng.uniform(0.4, 1)
        return x, 0.4
    if kind == "puff":
        return noise(rng, 0.3, 120, 900, 0.02, 0.08), 0.35
    if kind == "boing":                              # spring: bent pitch with a fast wobble
        return glide(140, 330, 0.32, 0.12, vib=0.25, vib_hz=22), 0.5
    if kind in ("lub", "dub"):
        x = glide(85 if kind == "lub" else 100, 50, 0.16, 0.05, attack=0.004)
        return x, 0.9 if kind == "lub" else 0.65
    if kind == "rattle":
        return mix(glide(950, 800, 0.03, 0.008), 0.5 * noise(rng, 0.03, 2000, 6000, 0.0005, 0.005)), 0.4
    if kind == "creak":
        return glide(260, 210, 0.18, 0.08, attack=0.03), 0.2
    if kind == "whoomp":
        return glide(130, 70, 0.28, 0.1, attack=0.03), 0.6
    if kind == "tick":
        return glide(2200, 2000, 0.02, 0.005), 0.3
    if kind == "blip":
        return glide(660 * 1.122 ** var, 680 * 1.122 ** var, 0.05, 0.025), 0.35
    if kind == "ding":                               # major third: "done"
        return mix(partials(1046.5, [1, 2.0], 0.8, 0.25), partials(1318.5, [1, 2.0], 0.8, 0.25)), 0.25
    if kind == "buzz":                               # two low square-ish notes: "nope"
        t = _t(0.14)
        a = np.sign(np.sin(2 * np.pi * 180 * t)) * env(0.14, 0.005, 0.08)
        b = np.sign(np.sin(2 * np.pi * 140 * t)) * env(0.14, 0.005, 0.08)
        return band(np.concatenate([a, b]), 80, 2500) * np.concatenate([env(0.14, 0.005, 0.08)] * 2), 0.35
    if kind == "tap":
        return mix(glide(400, 1000, 0.06, 0.02), 0.4 * noise(rng, 0.03, 2000, 7000, 0.0005, 0.004)), 0.5
    if kind == "bell":                               # struck bell: inharmonic partials
        return partials(880, [1, 2.0, 2.4, 3.0, 4.2], 0.9, 0.4), 0.22 if var else 0.3
    if kind == "thunk":
        return mix(glide(300, 110, 0.1, 0.035), 0.3 * noise(rng, 0.03, 200, 2000, 0.001, 0.01)), 0.6
    if kind == "fill":                               # rising tone while the bar fills (0.7 of a cycle)
        dur = 0.7 / 0.5
        x = glide(330, 660, dur, 99, attack=0.05)
        return x * np.minimum(1, (dur - _t(dur)) / 0.05), 0.08
    if kind == "done":
        return partials(1318.5, [1, 2.0], 0.4, 0.12), 0.25
    if kind == "click":
        return mix(glide(1500 if var else 1100, 1300 if var else 950, 0.015, 0.004), 0.4 * noise(rng, 0.004, 2000, 9000, 0.0002, 0.001)), 0.5
    if kind == "key":
        return noise(rng, 0.03, 1800, 7000, 0.0005, 0.006), 0.3
    raise ValueError(kind)


def motion_bed(cfg, rng):
    """Continuous sounds for spin / float / flicker, as a full 2 s loop."""
    t = np.arange(N) / SR
    p = (cfg["speed_hz"] * t + cfg.get("motion_phase", 0.0)) % 1.0
    m = cfg["motion"]
    if m == "spin":                                  # whoosh, loudest mid-turn
        return band(rng.standard_normal(N), 300, 2500) * (0.15 + 0.85 * np.sin(np.pi * p) ** 2) * 0.12
    if m == "float":                                 # soft airy hum rising and falling with the float
        return (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t)) * (0.5 + 0.5 * np.sin(2 * np.pi * p)) * 0.03
    if m == "flicker":                               # low fire roar following the flame's stretch
        return band(rng.standard_normal(N), 60, 500) * (0.6 + 0.4 * np.sin(2 * np.pi * p)) * 0.18
    return np.zeros(N)


def render_audio(cfg):
    """cfg (a clip's <id>.json) -> (2.0 s float32 loop at SR, event list)."""
    rng = np.random.default_rng(cfg["seed"])          # noise texture only; never touches timing
    buf = motion_bed(cfg, rng) if cfg["motion"] in MOTION_BEDS else np.zeros(N)
    ev = events(cfg)
    for t, kind, var in ev:
        x, gain = sound(kind, var, rng)
        idx = (int(round(t * SR)) + np.arange(len(x))) % N    # wrap past 2.0 s back to the start: seamless loop
        np.add.at(buf, idx, gain * x)
    peak = np.abs(buf).max()
    if peak > 0.89:                                   # keep under -1 dBFS; quieter clips stay quieter
        buf *= 0.89 / peak
    return buf.astype(np.float32), ev


# ---------- files ----------

def write_wav(path, x, repeat=1):
    pcm = (np.clip(np.tile(x, repeat), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def mux(clip_mp4, wav, out_mp4, loops):
    """16 frames (frame 17 repeats frame 1) looped `loops` times, with the matching audio; 512 px for viewing."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_mp4), "-i", str(wav),
         "-vf", f"trim=end_frame=16,setpts=N/{G.FPS}/TB,loop=loop={loops - 1}:size=16:start=0,"
                "scale=512:512:flags=neighbor,format=yuv420p",
         "-r", str(G.FPS), "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out_mp4)],
        check=True)


def manifest():
    """[(id, path of its .json)] for every clip."""
    return [(r["id"], G.DATA / r["path"].replace(".mp4", ".json")) for r in map(json.loads, open(G.DATA / "manifest.jsonl"))]


def video_frames(mp4):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, G.SIZE, G.SIZE, 3).astype(np.int16)


def reproduces(cfg):
    """Re-draw the clip from its spec with today's generator; mean pixel error vs the saved mp4 (h264 adds ~1-2)."""
    if cfg["category"] == "ui":
        art = G.ui_art(cfg)
    else:
        from PIL import Image
        art = Image.open(G.ASSETS / f"{cfg['art']}.png").convert("RGBA")
    phases = [random.Random(cfg["seed"]).uniform(0, 1) for _ in range(2)]
    mine = np.stack([np.asarray(G.render_frame(cfg, i / G.FPS, art, phases), np.int16) for i in range(0, G.FRAMES, 4)])
    theirs = video_frames(G.DATA / "clips" / cfg["category"] / cfg["subcategory"] / f"{cfg['id']}.mp4")[::4]
    return float(np.abs(mine - theirs).mean())


def pick_sample(n, rng):
    """Cover every effect, motion and UI element if n allows: each pick is the clip with the most still-uncovered
    tags (from 4,000 random recent clips), then any slots left are filled at random."""
    ids = manifest()
    rng.shuffle(ids)
    pool = [c for c in (json.loads(p.read_text()) for _, p in ids[:4000]) if c.get("generator_version", "0") >= "0.4"]
    tags = lambda c: {("effect", c["effect"]), ("motion", c["motion"])} | ({("ui", c["subcategory"])} if c["category"] == "ui" else set())
    want = {("effect", e) for e in G.EFFECTS} | {("motion", m) for m in G.MOTIONS} | {("ui", u) for u in G.UI}
    chosen = []
    while len(chosen) < n and want:
        best = max(pool, key=lambda c: len(tags(c) & want))
        if not tags(best) & want:
            break
        want -= tags(best)
        chosen.append(best)
        pool.remove(best)
    chosen += pool[:n - len(chosen)]
    return chosen, want


PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bhaav-E sound sample</title>
<style>
:root{--bg:#f7f6f2;--fg:#1d1d1f;--mute:#6b6b70;--card:#fff;--line:#e3e1da;--acc:#d9480f}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececec;--mute:#9a9aa2;--card:#1d1d21;--line:#2c2c31;--acc:#ff8a4c}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:24px 16px}
h1{font-size:24px;margin:0 0 4px}p{color:var(--mute);margin:0 0 20px;max-width:70ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
video{width:100%;border-radius:6px;display:block;background:#000}
.cap{font-size:13px;margin:8px 0 6px}.tags{font:12px ui-monospace,monospace;color:var(--mute)}
.bar{position:relative;height:18px;border-top:1px solid var(--line);margin-top:8px}
.bar i{position:absolute;top:3px;width:2px;height:12px;background:var(--acc)}
.bar b{position:absolute;top:0;width:2px;height:18px;background:var(--fg);opacity:.5}
</style><main><h1>Bhaav-E: sound sample</h1>
<p>Each clip plays its 2-second loop 3 times with sound made by <code>generator/bhaav_audio.py</code> from the clip's
spec alone. Orange ticks are the scheduled sound times (0 to 2 s); the grey line is the playhead. Click a clip for sound.</p>
<div class="grid" id="g"></div></main>
<script>
fetch('index.json').then(r=>r.json()).then(rows=>{for(const r of rows){const d=document.createElement('div');d.className='c';
d.innerHTML=`<video src="${r.id}.mp4" controls loop playsinline preload="metadata"></video><div class="cap">${r.caption}</div>
<div class="tags">${r.ui?'ui: '+r.ui+' · ':''}motion: ${r.motion} · effect: ${r.effect} · ${r.events.length} sounds</div>
<div class="bar">${r.events.map(e=>`<i style="left:${e[0]/2*100}%" title="${e[0]} s ${e[1]}"></i>`).join('')}<b></b></div>`;
const v=d.querySelector('video'),b=d.querySelector('b');
v.addEventListener('timeupdate',()=>b.style.left=(v.currentTime%2)/2*100+'%');
document.getElementById('g').append(d)}});
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    if a.cmd == "check":
        by_ver = {}
        for _, path in random.Random(1).sample(manifest(), a.n):
            cfg = json.loads(path.read_text())
            try:
                err = reproduces(cfg)
            except KeyError:                          # very old specs (v0.1-0.2) lack fields today's code needs
                err = float("inf")
            by_ver.setdefault(cfg.get("generator_version", "?"), []).append(err)
        for v, errs in sorted(by_ver.items()):
            errs = np.array(errs)
            print(f"v{v}: {len(errs)} clips, mean pixel error median {np.median(errs):.2f}, max {errs.max():.2f}, "
                  f"over 3: {(errs > 3).sum()}")
        return

    OUT.mkdir(exist_ok=True)
    chosen, missing = pick_sample(a.n, random.Random(a.seed))
    print(f"{len(chosen)} clips; not covered: {sorted(missing) or 'nothing'}")
    index = []
    for cfg in chosen:
        x, ev = render_audio(cfg)
        cid = cfg["id"]
        write_wav(OUT / f"{cid}.wav", x)
        write_wav(OUT / "tmp3.wav", x, repeat=3)
        mux(G.DATA / "clips" / cfg["category"] / cfg["subcategory"] / f"{cid}.mp4", OUT / "tmp3.wav", OUT / f"{cid}.mp4", 3)
        caption = (G.DATA / "clips" / cfg["category"] / cfg["subcategory"] / f"{cid}.txt").read_text().splitlines()[0]
        index.append({"id": cid, "caption": caption, "motion": cfg["motion"], "effect": cfg["effect"],
                      "ui": cfg["subcategory"] if cfg["category"] == "ui" else None,
                      "pixel_error": round(reproduces(cfg), 2), "events": ev})
        print(f"{cid:42s} {len(ev):3d} events  {caption}")
    (OUT / "tmp3.wav").unlink()
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    (OUT / "index.html").write_text(PAGE)


if __name__ == "__main__":
    main()
