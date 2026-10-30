"""Bhaav Tiny clip generator (our code).

Renders sticker clips from a settings dict and writes the four files
described in data/README.md: <id>.mp4, <id>.txt, <id>.json, <id>.png.

Every clip: 256x256, 17 frames, 8 fps, frame 17 == frame 1 (loops).
All motion and effects repeat with a period that divides 2 s, so the loop is exact.

Usage:
  one clip:   .venv/bin/python generator/bhaav_gen.py one --subject crying --motion bounce --effect tears
              .venv/bin/python generator/bhaav_gen.py one --category animated-emoji --subject fire --motion flicker --effect embers
  a batch:    .venv/bin/python generator/bhaav_gen.py batch --batch b0001 --n 100 --seed 1
              .venv/bin/python generator/bhaav_gen.py batch --category animated-emoji --batch e0001 --n 100 --seed 1001
  fill-in:    .venv/bin/python generator/bhaav_gen.py fill --plan data/r2/plan_0.json --batch f0000 --seed 9000
              (r2 rebalancing: exact subject + effect per clip; motion and accent drawn as usual but never a held-out combo)
"""

import argparse
import fcntl
import json
import math
import random
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

GENERATOR_VERSION = "0.6"   # 0.6: fill command for r2 rebalancing (0.5: ui drawn in code; 0.4: animated-emoji)
ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "generator" / "assets" / "fluent-emoji"
DATA = ROOT / "data"

SIZE = 256            # output size
SS = 2                # supersampling factor for smooth edges
FRAMES = 17
FPS = 8
DURATION = (FRAMES - 1) / FPS   # 2.0 s; frame 17 sits at t = 2.0 == t = 0

# ---------- what we can draw ----------
# arts: (file stem, caption name). eyes: where tears start, as fractions of the art image.
EMOTIONS = {
    "laugh": {
        "arts": [("face_with_tears_of_joy_3d", "laughing emoji with tears of joy"),
                 ("rolling_on_the_floor_laughing_3d", "rolling on the floor laughing emoji")],
        "words": ["laughing", "LOL", "cracking up", "haha"],
        "effects": ["tears-joy", "sparkles", "none"],
    },
    "crying": {
        "arts": [("crying_face_3d", "crying emoji"), ("loudly_crying_face_3d", "loudly crying emoji")],
        "words": ["crying", "sad", "upset", "in tears"],
        "effects": ["tears", "none"],
    },
    "heart-eyes": {
        "arts": [("smiling_face_with_heart-eyes_3d", "heart-eyes emoji")],
        "words": ["in love", "heart eyes", "adoring", "love it"],
        "effects": ["hearts", "sparkles", "none"],
    },
    "facepalm": {
        "arts": [("person_facepalming_3d_default", "facepalming emoji")],
        "words": ["facepalm", "embarrassed", "can't believe it", "ugh"],
        "effects": ["sweat", "none"],
    },
    "angry": {
        "arts": [("angry_face_3d", "angry emoji"), ("pouting_face_3d", "red furious emoji")],
        "words": ["angry", "mad", "furious", "annoyed"],
        "effects": ["steam", "none"],
    },
    "shocked": {
        "arts": [("astonished_face_3d", "astonished emoji"), ("face_screaming_in_fear_3d", "screaming in shock emoji")],
        "words": ["shocked", "surprised", "omg", "stunned"],
        "effects": ["sweat", "sparkles", "none"],
    },
    "wink": {
        "arts": [("winking_face_3d", "winking emoji")],
        "words": ["winking", "cheeky", "playful", "flirty"],
        "effects": ["sparkles", "hearts", "none"],
    },
    "thinking": {
        "arts": [("thinking_face_3d", "thinking emoji")],
        "words": ["thinking", "hmm", "wondering", "pondering"],
        "effects": ["thought", "none"],
    },
    "sleepy": {
        "arts": [("sleeping_face_3d", "sleeping emoji")],
        "words": ["sleepy", "sleeping", "tired", "bored"],
        "effects": ["zzz", "none"],
    },
    "cool": {
        "arts": [("smiling_face_with_sunglasses_3d", "cool emoji with sunglasses")],
        "words": ["cool", "chill", "confident", "swag"],
        "effects": ["sparkles", "none"],
    },
}

# Objects and symbols. "motions" limits which motions fit; "anchor" is where bursts start (u, v on the art).
OBJECTS = {
    "fire": {
        "arts": [("fire_3d", "fire emoji")],
        "words": ["fire", "lit", "on fire", "hot"],
        "effects": ["embers", "sparkles", "none"],
        "motions": ["flicker", "pulse", "bounce", "wobble", "float"],
        "anchor": (0.5, 0.05),
    },
    "heart": {
        "arts": [("red_heart_3d", "red heart emoji")],
        "words": ["love", "heart", "love you", "pyaar"],
        "effects": ["hearts", "sparkles", "none"],
        "motions": ["heartbeat", "pulse", "bounce", "float", "wobble"],
        "anchor": (0.5, 0.05),
    },
    "party": {
        "arts": [("party_popper_3d", "party popper emoji")],
        "words": ["party", "celebration", "congrats", "let's go"],
        "effects": ["confetti", "sparkles", "none"],
        "motions": ["shake", "wobble", "bounce", "pulse"],
        "anchor": (0.78, 0.18),
    },
    "thumbs-up": {
        "arts": [("thumbs_up_3d_default", "thumbs up emoji")],
        "words": ["thumbs up", "ok", "done", "great job"],
        "effects": ["sparkles", "none"],
        "motions": ["bounce", "pulse", "wobble", "shake", "float"],
        "anchor": (0.5, 0.05),
    },
    "star": {
        "arts": [("glowing_star_3d", "glowing star emoji")],
        "words": ["star", "superstar", "amazing", "shining"],
        "effects": ["sparkles", "none"],
        "motions": ["spin", "pulse", "float", "bounce", "wobble"],
        "anchor": (0.5, 0.05),
    },
    "hundred": {
        "arts": [("hundred_points_3d", "100 points emoji")],
        "words": ["100", "perfect", "totally", "full marks"],
        "effects": ["sparkles", "confetti", "none"],
        "motions": ["pulse", "bounce", "shake", "wobble"],
        "anchor": (0.5, 0.0),
    },
    "clap": {
        "arts": [("clapping_hands_3d_default", "clapping hands emoji")],
        "words": ["clap", "applause", "well done", "bravo"],
        "effects": ["sparkles", "confetti", "none"],
        "motions": ["shake", "pulse", "bounce", "wobble"],
        "anchor": (0.5, 0.0),
    },
    "rocket": {
        "arts": [("rocket_3d", "rocket emoji")],
        "words": ["rocket", "launch", "to the moon", "fast"],
        "effects": ["exhaust", "sparkles", "none"],
        "motions": ["shake", "float", "bounce", "wobble"],
        "anchor": (0.12, 0.88),
    },
    "trophy": {
        "arts": [("trophy_3d", "gold trophy emoji")],
        "words": ["trophy", "winner", "champion", "first place"],
        "effects": ["sparkles", "confetti", "none"],
        "motions": ["bounce", "pulse", "wobble", "float"],   # no spin: upside-down trophy looks wrong
        "anchor": (0.5, 0.0),
    },
    "idea": {
        "arts": [("light_bulb_3d", "light bulb emoji")],
        "words": ["idea", "bright idea", "eureka", "got it"],
        "effects": ["sparkles", "none"],
        "motions": ["pulse", "wobble", "bounce", "float", "flicker"],
        "anchor": (0.5, 0.05),
    },
}

BACKGROUNDS = {
    "light blue": "#cfe8ff", "pink": "#ffd6e7", "mint green": "#d4f5e3", "lavender": "#e4dcff",
    "peach": "#ffe0c7", "pale yellow": "#fff4c2", "white": "#ffffff", "light grey": "#ececec",
    "sky blue": "#a9d8ff", "teal": "#9ee6e0", "dark navy": "#1f2a44", "black": "#111111",
}


# ---------- motion: (dx, dy, sx, sy, angle_deg) at time t; p is the 0..1 cycle position ----------

def motion_bounce(p, amp=28):
    h = abs(math.sin(math.pi * p))                  # 0 on the ground, 1 at the top
    squash = (1 - h) ** 3                           # only near the ground
    return 0.0, amp / 2 - amp * h, 1 + 0.06 * squash, 1 - 0.08 * squash, 0.0


def motion_wobble(p):
    return 0.0, 0.0, 1.0, 1.0, 12 * math.sin(2 * math.pi * p)


def motion_pulse(p):
    s = 1 + 0.08 * math.sin(2 * math.pi * p)
    return 0.0, 0.0, s, s, 0.0


def motion_float(p):
    return 0.0, 10 * math.sin(2 * math.pi * p), 1.0, 1.0, 3 * math.sin(2 * math.pi * p + 1)


def motion_shake(p):
    return 9 * math.sin(2 * math.pi * p), 0.0, 1.0, 1.0, 0.0


def motion_still(p):
    return 0.0, 0.0, 1.0, 1.0, 0.0


def motion_spin(p):
    return 0.0, 0.0, 1.0, 1.0, -360 * p             # one full clockwise turn per cycle


def motion_heartbeat(p):
    """Two quick beats, then a rest: lub-dub."""
    beat = max(math.exp(-((p - 0.1) / 0.07) ** 2), 0.7 * math.exp(-((p - 0.35) / 0.07) ** 2))
    s = 1 + 0.15 * beat
    return 0.0, 0.0, s, s, 0.0


def motion_flicker(p):
    """Stretch tall and thin, then short and wide, like a flame."""
    k = math.sin(2 * math.pi * p)
    return 0.0, 0.0, 1 - 0.05 * k, 1 + 0.07 * k, 3 * math.sin(4 * math.pi * p)


# name: (function, allowed speeds in Hz, verb, adjective). Speeds keep >= 4 frames per cycle at 8 fps.
MOTIONS = {
    "bounce": (motion_bounce, [0.5, 1.0], "bouncing", "bouncy"),
    "wobble": (motion_wobble, [0.5, 1.0], "wobbling side to side", "wobbly"),
    "pulse": (motion_pulse, [0.5, 1.0], "pulsing", "pulsing"),
    "float": (motion_float, [0.5], "floating gently", "floaty"),
    "shake": (motion_shake, [1.0, 2.0], "shaking", "shaky"),
    "still": (motion_still, [0.5], "staying still", "still"),
    "spin": (motion_spin, [0.5], "spinning around", "spinning"),
    "heartbeat": (motion_heartbeat, [1.0], "beating like a heart", "beating"),
    "flicker": (motion_flicker, [1.0, 2.0], "flickering", "flickering"),
}
BASE_MOTIONS = ["bounce", "wobble", "pulse", "float", "shake", "still"]   # reactions use these

# category: (subjects, noun used in captions)
CATEGORIES = {
    "reactions": (EMOTIONS, "reaction emoji"),
    "animated-emoji": (OBJECTS, "animated emoji"),
}


def face_transform(cfg, t):
    """Centre, size and rotation of the emoji at time t (output pixels)."""
    fn = MOTIONS[cfg["motion"]][0]
    dx, dy, sx, sy, ang = fn((cfg["speed_hz"] * t + cfg["motion_phase"]) % 1.0)
    w, h = cfg["emoji_px"] * sx, cfg["emoji_px"] * sy
    cx = SIZE / 2 + cfg["offset"][0] + dx
    cy = SIZE / 2 + cfg["offset"][1] + dy + (cfg["emoji_px"] - h) / 2   # squash stays anchored at the bottom
    return cx, cy, w, h, ang


def to_screen(cfg, t, u, v):
    """A point on the art (u, v in 0..1) to output pixels, following the face's motion."""
    cx, cy, w, h, ang = face_transform(cfg, t)
    x, y = (u - 0.5) * w, (v - 0.5) * h
    a = math.radians(-ang)                          # PIL rotates counter-clockwise for positive angles
    return cx + x * math.cos(a) - y * math.sin(a), cy + x * math.sin(a) + y * math.cos(a)


# ---------- effects: drawn on top of the face, in supersampled pixels ----------

def draw_drop(draw, x, y, r, alpha, colour=(90, 170, 255)):
    fill = colour + (alpha,)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    draw.polygon([(x - r * 0.92, y - r * 0.35), (x + r * 0.92, y - r * 0.35), (x, y - r * 2.3)], fill=fill)
    draw.ellipse([x - r * 0.45, y - r * 0.5, x - r * 0.05, y - r * 0.1], fill=(230, 245, 255, alpha))


def draw_heart(draw, x, y, r, alpha):
    fill = (255, 70, 110, alpha)
    draw.ellipse([x - r, y - r * 0.6, x, y + r * 0.4], fill=fill)
    draw.ellipse([x, y - r * 0.6, x + r, y + r * 0.4], fill=fill)
    draw.polygon([(x - r * 0.97, y), (x + r * 0.97, y), (x, y + r * 1.15)], fill=fill)


def draw_sparkle(draw, x, y, r, alpha):
    k = r * 0.22
    pts = [(x, y - r), (x + k, y - k), (x + r, y), (x + k, y + k), (x, y + r), (x - k, y + k), (x - r, y), (x - k, y - k)]
    draw.polygon(pts, fill=(255, 230, 90, alpha))


def draw_z(draw, x, y, r, alpha, colour):
    w = max(2, int(r * 0.28))
    draw.line([(x - r, y - r), (x + r, y - r), (x - r, y + r), (x + r, y + r)], fill=colour + (alpha,), width=w, joint="curve")


def fade(age, life, fade_in=0.12, fade_out=0.35):
    return int(255 * max(0.0, min(1.0, age / (fade_in * life), (life - age) / (fade_out * life))))


def ages(t, phase, every, life):
    """Ages of all drops alive at time t, for a stream emitting one drop every `every` seconds."""
    start = round((t - phase) % every, 6) % every   # rounding: t=2.0 must give exactly the same ages as t=0
    return np.arange(start, life, every)


def effect_tears(cfg, t, draw, rng_ph):
    for side, (u, v) in enumerate([(0.28, 0.6), (0.72, 0.6)]):
        for age in ages(t, rng_ph[side], 0.5, 0.75):
            ex, ey = to_screen(cfg, t - age, u, v)  # where the face was when the drop left
            x = ex + (-22 if side == 0 else 22) * age
            y = ey + 0.5 * 430 * age * age
            r = 7 * min(1.0, 0.45 + age / 0.15 * 0.55)
            draw_drop(draw, x * SS, y * SS, r * SS, fade(age, 0.75, fade_in=0.01))


def effect_tears_joy(cfg, t, draw, rng_ph):
    """Tears of joy spray outwards in arcs from both sides."""
    for side, (u, v) in enumerate([(0.1, 0.45), (0.9, 0.45)]):
        for age in ages(t, rng_ph[side], 0.25, 0.6):
            ex, ey = to_screen(cfg, t - age, u, v)
            x = ex + (-70 if side == 0 else 70) * age
            y = ey - 90 * age + 0.5 * 500 * age * age
            draw_drop(draw, x * SS, y * SS, 6 * SS, fade(age, 0.6, fade_in=0.01))


def effect_hearts(cfg, t, draw, rng_ph):
    for i, u in enumerate([0.15, 0.5, 0.85]):
        for age in ages(t, rng_ph[i % 2] + i * 0.33, 1.0, 1.0):
            sx, sy = to_screen(cfg, t - age, u, 0.15)
            x = sx + 8 * math.sin(2 * math.pi * age + i)
            y = sy - 70 * age
            draw_heart(draw, x * SS, y * SS, (6 + 6 * age) * SS, fade(age, 1.0))


def effect_sparkles(cfg, t, draw, rng_ph):
    spots = [(-0.05, 0.1), (1.05, 0.2), (0.95, 0.95), (0.0, 0.85)]
    for i, (u, v) in enumerate(spots):
        twinkle = max(0.0, math.sin(2 * math.pi * (t + rng_ph[i % 2] + i * 0.25)))
        if twinkle > 0.05:
            x, y = to_screen(cfg, t, u, v)
            draw_sparkle(draw, x * SS, y * SS, 12 * twinkle * SS, int(255 * twinkle))


def effect_sweat(cfg, t, draw, rng_ph):
    """One sweat drop slides down the forehead, then falls."""
    for age in ages(t, rng_ph[0], 1.0, 1.0):
        x, y = to_screen(cfg, t, 0.8, 0.12)
        slide = min(age, 0.6) / 0.6
        fall = max(0.0, age - 0.6)
        draw_drop(draw, x * SS, (y + 22 * slide + 0.5 * 500 * fall * fall) * SS, 8 * SS, fade(age, 1.0), (120, 200, 255))


def effect_steam(cfg, t, draw, rng_ph):
    for side, u in enumerate([0.12, 0.88]):
        for age in ages(t, rng_ph[side], 0.25, 0.75):
            sx, sy = to_screen(cfg, t - age, u, 0.05)
            x = sx + (-25 if side == 0 else 25) * age
            y = sy - 60 * age
            r = (5 + 14 * age) * SS
            a = fade(age, 0.75) * 3 // 4
            colour = (245, 245, 245) if cfg["dark_background"] else (140, 140, 150)   # visible on any background
            draw.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=colour + (a,))


def effect_zzz(cfg, t, draw, rng_ph):
    colour = (255, 255, 255) if cfg["dark_background"] else (70, 90, 160)
    for age in ages(t, rng_ph[0], 0.5, 1.0):
        sx, sy = to_screen(cfg, t - age, 1.0, 0.05)   # beside the Zzz already painted on the art
        x = sx + 20 * age + 4 * math.sin(4 * math.pi * age)
        y = sy - 35 * age
        draw_z(draw, x * SS, y * SS, (5 + 6 * age) * SS, fade(age, 1.0), colour)


def effect_thought(cfg, t, draw, rng_ph):
    """Three thought bubbles appear one after another above the head, then fade together."""
    colour = (255, 255, 255)
    p = (t + rng_ph[0]) % 1.0
    for i, (u, v, r) in enumerate([(0.9, 0.05, 4), (1.0, -0.12, 7), (1.05, -0.32, 11)]):
        shown = p - i * 0.2
        if 0 <= shown and p < 0.9:
            x, y = to_screen(cfg, t, u, v)
            rr = r * min(1.0, shown / 0.1) * SS
            draw.ellipse([x * SS - rr, y * SS - rr, x * SS + rr, y * SS + rr], fill=colour + (235,), outline=(150, 150, 170, 235), width=SS)


CONFETTI_COLOURS = [(255, 80, 120), (255, 200, 60), (80, 200, 255), (120, 220, 120), (190, 120, 255), (255, 140, 60)]


def effect_confetti(cfg, t, draw, rng_ph):
    """A burst of 14 paper bits from the anchor once per second; they arc out, tumble and fall."""
    u, v = cfg["anchor"]
    for age in ages(t, rng_ph[0], 1.0, 1.0):
        ox, oy = to_screen(cfg, t - age, u, v)
        for k in range(14):
            ang = math.radians(-160 + 140 * ((k * 0.618 + rng_ph[1]) % 1.0))   # spread over the upper half
            speed = 90 + 50 * ((k * 0.37) % 1.0)
            x = ox + math.cos(ang) * speed * age
            y = oy + math.sin(ang) * speed * age + 0.5 * 260 * age * age
            spin = 6 * age + k
            w, h = 4.5 * SS, 2.2 * SS
            c, s = math.cos(spin), math.sin(spin)
            pts = [(x * SS + c * dx - s * dy, y * SS + s * dx + c * dy) for dx, dy in [(-w, -h), (w, -h), (w, h), (-w, h)]]
            draw.polygon(pts, fill=CONFETTI_COLOURS[k % len(CONFETTI_COLOURS)] + (fade(age, 1.0, fade_in=0.01),))


def effect_embers(cfg, t, draw, rng_ph):
    """Small glowing sparks rise from the flame tip and drift."""
    u, v = cfg["anchor"]
    for k in range(4):
        for age in ages(t, rng_ph[0] + k * 0.13, 0.5, 0.8):
            ox, oy = to_screen(cfg, t - age, u + (k - 1.5) * 0.12, v + 0.1)
            x = ox + 10 * math.sin(2 * math.pi * age * 1.5 + k)
            y = oy - 60 * age
            r = (3.2 - 2 * age) * SS
            colour = (255, 210, 80) if k % 2 else (255, 140, 40)
            draw.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=colour + (fade(age, 0.8),))


def effect_exhaust(cfg, t, draw, rng_ph):
    """Smoke puffs leave the rocket's tail and drift down-left, growing and fading."""
    u, v = cfg["anchor"]
    colour = (235, 235, 240) if cfg["dark_background"] else (150, 150, 160)
    for age in ages(t, rng_ph[0], 0.25, 0.75):
        ox, oy = to_screen(cfg, t - age, u, v)
        x, y = ox - 45 * age, oy + 45 * age
        r = (4 + 12 * age) * SS
        draw.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=colour + (fade(age, 0.75, fade_in=0.01) * 3 // 4,))
        fr = max(0.0, 5 - 30 * age) * SS                  # small flame right at the nozzle
        if fr > 0:
            draw.ellipse([x * SS - fr, y * SS - fr, x * SS + fr, y * SS + fr], fill=(255, 160, 50, 230))


def effect_none(cfg, t, draw, rng_ph):
    pass


# name: (function, caption phrase, short phrase)
EFFECTS = {
    "tears": (effect_tears, "tears falling", "tears"),
    "tears-joy": (effect_tears_joy, "tears of joy flying out", "tears of joy"),
    "hearts": (effect_hearts, "little hearts floating up", "hearts"),
    "sparkles": (effect_sparkles, "sparkles twinkling around it", "sparkles"),
    "sweat": (effect_sweat, "a sweat drop sliding down", "sweat drop"),
    "steam": (effect_steam, "steam puffing from its head", "steam"),
    "zzz": (effect_zzz, "Zzz floating up", "Zzz"),
    "thought": (effect_thought, "thought bubbles popping up", "thought bubbles"),
    "confetti": (effect_confetti, "confetti bursting out", "confetti"),
    "embers": (effect_embers, "sparks rising from the flame", "sparks"),
    "exhaust": (effect_exhaust, "smoke puffing from its tail", "smoke trail"),
    "none": (effect_none, "", ""),
}


# ---------- UI elements: drawn in code each frame, no art file ----------
# Each function gets the art size S (supersampled px), the cycle position p (0..1, loops),
# the accent colour, and the track colour; it returns an S x S RGBA image.

TRACK = (205, 205, 215)


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def with_alpha(img, a):
    """Multiply the whole image's opacity by a (0..1)."""
    if a < 1:
        img.putalpha(img.getchannel("A").point(lambda v: int(v * clamp01(a))))
    return img


def partial_path(d, pts, q, colour, w):
    """Draw the first q (0..1) of a polyline, by length."""
    segs = list(zip(pts, pts[1:]))
    total = sum(math.dist(a, b) for a, b in segs)
    left = q * total
    for a, b in segs:
        L = math.dist(a, b)
        if left <= 0:
            break
        f = min(1.0, left / L)
        end = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        d.line([a, end], fill=colour, width=w)
        for x, y in (a, end):                       # round caps
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=colour)
        left -= L


def heart_shape(d, cx, cy, r, fill):
    d.ellipse([cx - r, cy - r * 0.75, cx + 0.04 * r, cy + r * 0.3], fill=fill)
    d.ellipse([cx - 0.04 * r, cy - r * 0.75, cx + r, cy + r * 0.3], fill=fill)
    d.polygon([(cx - r * 0.95, cy - r * 0.05), (cx + r * 0.95, cy - r * 0.05), (cx, cy + r * 1.05)], fill=fill)


def ui_spinner(S, p, col):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, m = int(S * 0.09), S * 0.14
    d.ellipse([m, m, S - m, S - m], outline=TRACK + (255,), width=w)
    start = 360 * p - 90
    length = 70 + 160 * math.sin(math.pi * p) ** 2   # arc grows and shrinks as it turns
    d.arc([m, m, S - m, S - m], start, start + length, fill=col, width=w)
    return img


def ui_dots(S, p, col):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = S * 0.1
    for k in range(3):
        lift = max(0.0, math.sin(2 * math.pi * (p - k * 0.15))) * S * 0.16
        x, y = S * (0.25 + 0.25 * k), S * 0.55 - lift
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return img


def _ring_then_mark(S, p, col, strokes):
    """Ring draws in, then the mark's strokes draw one after another, hold, fade out."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, m = int(S * 0.08), S * 0.1
    ring = clamp01(p / 0.3)
    d.ellipse([m, m, S - m, S - m], fill=col[:3] + (int(40 * ring),))
    if ring > 0:
        d.arc([m, m, S - m, S - m], -90, -90 + 360 * ring, fill=col, width=w)
    q = clamp01((p - 0.3) / 0.25)
    for k, pts in enumerate(strokes):
        part = clamp01(q * len(strokes) - k)
        if part > 0:
            partial_path(d, [(x * S, y * S) for x, y in pts], part, col, w)
    return with_alpha(img, 1 - clamp01((p - 0.85) / 0.15))


def ui_success(S, p, col):
    return _ring_then_mark(S, p, col, [[(0.3, 0.52), (0.44, 0.66), (0.71, 0.37)]])


def ui_error(S, p, col):
    return _ring_then_mark(S, p, col, [[(0.36, 0.36), (0.64, 0.64)], [(0.64, 0.36), (0.36, 0.64)]])


def ui_like(S, p, col):
    """Outline heart; tap: it pops and fills, little dots burst out; holds; fades back to outline."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pop = math.sin(math.pi * clamp01((p - 0.15) / 0.2))
    r = S * 0.3 * (1 + 0.25 * pop)
    cx, cy = S / 2, S * 0.5
    heart_shape(d, cx, cy, r, col)
    fill = clamp01((p - 0.15) / 0.05) * (1 - clamp01((p - 0.8) / 0.2))
    if fill < 1:                                     # erase the inside to leave an outline
        heart_shape(d, cx, cy, r * 0.8, col[:3] + (int(255 * fill),) if fill > 0 else (0, 0, 0, 0))
    burst = clamp01((p - 0.15) / 0.25)
    if 0 < burst < 1:
        for k in range(8):
            a = math.radians(k * 45 - 90)
            rr, dot = S * (0.3 + 0.17 * burst), S * 0.035 * (1 - burst)
            x, y = cx + math.cos(a) * rr, cy + math.sin(a) * rr
            d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=col)
    return img


def ui_bell(S, p, col):
    """Bell swings and settles; a red badge pops on, then fades before the loop restarts."""
    bell = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(bell)
    d.ellipse([S * 0.47, S * 0.12, S * 0.53, S * 0.18], fill=col)                    # top knob
    d.pieslice([S * 0.25, S * 0.16, S * 0.75, S * 0.66], 180, 360, fill=col)         # dome
    d.rectangle([S * 0.25, S * 0.41, S * 0.75, S * 0.66], fill=col)                  # body
    d.rounded_rectangle([S * 0.17, S * 0.62, S * 0.83, S * 0.72], radius=S * 0.05, fill=col)  # rim
    d.ellipse([S * 0.43, S * 0.72, S * 0.57, S * 0.84], fill=col)                    # clapper
    swing = 20 * math.sin(4 * math.pi * p) * clamp01(1 - p / 0.6)
    img = bell.rotate(swing, resample=Image.BICUBIC, center=(S * 0.5, S * 0.15))
    badge = math.sin(math.pi * clamp01((p - 0.05) / 0.2) / 2) * (1 - clamp01((p - 0.85) / 0.15))
    if badge > 0:
        d2, br = ImageDraw.Draw(img), S * 0.1 * min(1.0, badge * 1.15)
        d2.ellipse([S * 0.72 - br, S * 0.22 - br, S * 0.72 + br, S * 0.22 + br], fill=(239, 68, 68, int(255 * badge)))
    return img


def ui_download(S, p, col):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = int(S * 0.08)
    partial_path(d, [(S * 0.2, S * 0.62), (S * 0.2, S * 0.8), (S * 0.8, S * 0.8), (S * 0.8, S * 0.62)], 1, TRACK + (255,), w)
    y = S * (0.1 + 0.45 * smooth(p / 0.6))           # arrow drops into the tray
    a = clamp01(p / 0.1) * (1 - clamp01((p - 0.55) / 0.15))
    if a > 0:
        arrow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        da = ImageDraw.Draw(arrow)
        da.line([(S * 0.5, y), (S * 0.5, y + S * 0.25)], fill=col, width=w)
        da.polygon([(S * 0.33, y + S * 0.16), (S * 0.67, y + S * 0.16), (S * 0.5, y + S * 0.34)], fill=col)
        img.alpha_composite(with_alpha(arrow, a))
    return img


def ui_progress(S, p, col):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = S * 0.04, S * 0.96, S * 0.41, S * 0.59
    d.rounded_rectangle([x0, y0, x1, y1], radius=(y1 - y0) / 2, fill=TRACK + (255,))
    f = smooth(p / 0.7)
    if f > 0.02:
        fill = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(fill).rounded_rectangle([x0, y0, x0 + (y1 - y0) + (x1 - x0 - (y1 - y0)) * f, y1],
                                               radius=(y1 - y0) / 2, fill=col)
        img.alpha_composite(with_alpha(fill, 1 - clamp01((p - 0.85) / 0.15)))
    return img


def ui_toggle(S, p, col):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    on = smooth((p - 0.1) / 0.25) - smooth((p - 0.6) / 0.25)
    track = tuple(int(a + (b - a) * on) for a, b in zip(TRACK, col[:3])) + (255,)
    x0, x1, y0, y1 = S * 0.18, S * 0.82, S * 0.34, S * 0.66
    d.rounded_rectangle([x0, y0, x1, y1], radius=(y1 - y0) / 2, fill=track)
    r = (y1 - y0) * 0.4
    kx = x0 + (y1 - y0) / 2 + (x1 - x0 - (y1 - y0)) * on
    d.ellipse([kx - r, (y0 + y1) / 2 - r + S * 0.01, kx + r, (y0 + y1) / 2 + r + S * 0.01], fill=(0, 0, 0, 40))  # shadow
    d.ellipse([kx - r, (y0 + y1) / 2 - r, kx + r, (y0 + y1) / 2 + r], fill=(255, 255, 255, 255))
    return img


def ui_typing(S, p, col):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([S * 0.1, S * 0.3, S * 0.9, S * 0.68], radius=S * 0.19, fill=col)
    d.polygon([(S * 0.18, S * 0.6), (S * 0.32, S * 0.66), (S * 0.12, S * 0.8)], fill=col)   # tail
    r = S * 0.06
    for k in range(3):
        a = 0.35 + 0.65 * max(0.0, math.sin(2 * math.pi * (p - k * 0.17)))
        x = S * (0.32 + 0.18 * k)
        d.ellipse([x - r, S * 0.49 - r, x + r, S * 0.49 + r], fill=(255, 255, 255, int(255 * a)))
    return img


ACCENTS = {
    "green": (34, 197, 94), "teal": (20, 184, 166), "blue": (59, 130, 246), "purple": (139, 92, 246),
    "pink": (236, 72, 153), "red": (239, 68, 68), "orange": (249, 115, 22),
}
ALL_ACCENTS = list(ACCENTS)

# draw: the element's own animation. speeds: cycles per second for that animation (must divide 2 s).
# The element moves on its own, so the extra motion is usually "still".
UI_MOTIONS, UI_MOTION_WEIGHTS = ["still", "float", "pulse"], [3, 1, 1]
UI = {
    "spinner": {"draw": ui_spinner, "name": "loading spinner", "action": "spinning round and round",
                "words": ["loading", "please wait", "spinner", "buffering"], "accents": ALL_ACCENTS,
                "speeds": [0.5, 1.0], "effects": ["none"]},
    "dots-loader": {"draw": ui_dots, "name": "three-dot loader", "action": "with the dots bouncing one after another",
                    "words": ["loading", "wait", "dots loader", "processing"], "accents": ALL_ACCENTS,
                    "speeds": [0.5, 1.0], "effects": ["none"]},
    "success": {"draw": ui_success, "name": "success checkmark", "action": "drawing itself inside a circle",
                "words": ["done", "success", "completed", "tick"], "accents": ["green", "teal", "blue"],
                "speeds": [0.5], "effects": ["confetti", "sparkles", "none"]},
    "error": {"draw": ui_error, "name": "error cross", "action": "drawing itself inside a circle",
              "words": ["error", "failed", "wrong", "nope"], "accents": ["red", "orange", "pink"],
              "speeds": [0.5], "effects": ["none"]},
    "like": {"draw": ui_like, "name": "like heart button", "action": "being tapped and filling in with a pop",
             "words": ["like", "love it", "favourite", "liked"], "accents": ["red", "pink", "purple", "blue"],
             "speeds": [0.5], "effects": ["sparkles", "none"]},
    "bell": {"draw": ui_bell, "name": "notification bell", "action": "ringing as a red badge pops up",
             "words": ["notification", "new alert", "ding", "reminder"], "accents": ["blue", "purple", "orange", "teal", "green"],
             "speeds": [0.5], "effects": ["sparkles", "none"]},
    "download": {"draw": ui_download, "name": "download arrow", "action": "dropping down into a tray",
                 "words": ["download", "save", "downloading", "get it"], "accents": ALL_ACCENTS,
                 "speeds": [0.5, 1.0], "effects": ["none"]},
    "progress": {"draw": ui_progress, "name": "progress bar", "action": "filling up from left to right",
                 "words": ["progress", "loading bar", "almost there", "uploading"], "accents": ALL_ACCENTS,
                 "speeds": [0.5], "effects": ["none"]},
    "toggle": {"draw": ui_toggle, "name": "toggle switch", "action": "sliding on and then off",
               "words": ["toggle", "switch on", "settings", "on off"], "accents": ALL_ACCENTS,
               "speeds": [0.5], "effects": ["none"]},
    "typing": {"draw": ui_typing, "name": "typing indicator chat bubble", "action": "with its dots pulsing",
               "words": ["typing", "someone is typing", "chat", "replying"], "accents": ALL_ACCENTS,
               "speeds": [0.5, 1.0], "effects": ["none"]},
}
for _e in UI.values():
    _e.update(motions=UI_MOTIONS, motion_weights=UI_MOTION_WEIGHTS, anchor=(0.5, 0.1))
    _e.setdefault("start_phases", [0.0, 0.25, 0.5, 0.75])
for _k in ("success", "error"):                     # p = 0 is a blank frame for these; never start there
    UI[_k]["start_phases"] = [0.15, 0.4, 0.65]
CATEGORIES["ui"] = (UI, "UI animation")


def ui_art(cfg):
    """A function of t that draws this clip's UI element, used in place of an emoji image."""
    fn, S = UI[cfg["subcategory"]]["draw"], round(cfg["emoji_px"] * SS)
    col = ACCENTS[cfg["accent"]] + (255,)
    # rounding: t = 2.0 must land on exactly the same p as t = 0, or the loop check fails
    return lambda t: fn(S, round((cfg["ui_speed_hz"] * t + cfg["ui_phase"]) % 1.0, 6) % 1.0, col)


# ---------- rendering ----------

def render_frame(cfg, t, emoji, phases):
    canvas = Image.new("RGBA", (SIZE * SS, SIZE * SS), cfg["background"])
    cx, cy, w, h, ang = face_transform(cfg, t)
    art = emoji(t) if callable(emoji) else emoji     # UI elements are drawn fresh every frame
    face = art.resize((round(w * SS), round(h * SS)), Image.LANCZOS)
    if abs(ang) > 0.01:
        face = face.rotate(ang, resample=Image.BICUBIC, expand=True)
    canvas.alpha_composite(face, (round(cx * SS - face.width / 2), round(cy * SS - face.height / 2)))
    fx = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    EFFECTS[cfg["effect"]][0](cfg, t, ImageDraw.Draw(fx), phases)
    canvas.alpha_composite(fx)
    return canvas.resize((SIZE, SIZE), Image.LANCZOS).convert("RGB")


def captions(cfg, rng):
    """Line 1 is the full description; the others say the same thing in other words."""
    subjects, noun = CATEGORIES[cfg["category"]]
    e = subjects[cfg["subcategory"]]
    _, _, verb, adj = MOTIONS[cfg["motion"]]
    phrase, short = EFFECTS[cfg["effect"]][1:]
    colour = cfg["background_name"]
    with_fx = f", {phrase}" if phrase else ""
    if "action" in e:                                # UI: the element's own animation is the main verb
        verb = e["action"] + ("" if cfg["motion"] == "still" else f", {verb}")
        adj = "animated" if cfg["motion"] == "still" else adj
    article = "an" if cfg["art_name"][0] in "aeiou" else "a"
    main = f"{article} {cfg['art_name']} {verb}{with_fx}, {colour} background"
    word = rng.choice(e["words"])
    alts = [
        f"{word} sticker, {adj}" + (f", {short}" if short else ""),
        f"{cfg['art_name']}, {adj}" + (f", {short}" if short else "") + f", {colour} background, loops",
        f"{word} {noun} {verb} on a plain {colour} background" + (f" with {short}" if short else ""),
    ]
    rng.shuffle(alts)
    return [main] + alts[:2]


def next_id(category, sub):
    folder = DATA / "clips" / category / sub
    nums = [int(p.stem.rsplit("-", 1)[1]) for p in folder.glob(f"{category}-{sub}-*.json")]
    return f"{category}-{sub}-{max(nums, default=0) + 1:06d}"


@contextmanager
def data_lock():
    """Several batch processes may run at once; ids and the manifest are claimed one at a time."""
    with open(DATA / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def make_clip(cfg):
    rng = random.Random(cfg["seed"])
    folder = DATA / "clips" / cfg["category"] / cfg["subcategory"]
    folder.mkdir(parents=True, exist_ok=True)
    with data_lock():
        cfg["id"] = next_id(cfg["category"], cfg["subcategory"])
        (folder / f"{cfg['id']}.json").write_text("{}\n")   # claim the id; filled in below

    if cfg["category"] == "ui":
        emoji = ui_art(cfg)
    else:
        emoji = Image.open(ASSETS / f"{cfg['art']}.png").convert("RGBA")
    phases = [rng.uniform(0, 1) for _ in range(2)]   # effect timing, differs per seed

    frames = [render_frame(cfg, i / FPS, emoji, phases) for i in range(FRAMES)]
    loop_diff = np.abs(np.asarray(frames[0], int) - np.asarray(frames[-1], int)).max()
    assert loop_diff <= 1, f"{cfg['id']}: frame 17 != frame 1 (max pixel diff {loop_diff})"

    base = folder / cfg["id"]
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(f"{tmp}/{i:02d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%02d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-r", str(FPS), f"{base}.mp4"],
            check=True,
        )
    frames[0].save(f"{base}.png")
    base.with_suffix(".txt").write_text("\n".join(captions(cfg, rng)) + "\n")
    base.with_suffix(".json").write_text(json.dumps(cfg, indent=2) + "\n")

    with data_lock(), open(DATA / "manifest.jsonl", "a") as m:
        m.write(json.dumps({"id": cfg["id"], "path": str(base.relative_to(DATA)) + ".mp4",
                            "category": cfg["category"], "batch": cfg["batch"], "review": "pending"}) + "\n")
    return cfg["id"], frames[:-1]   # 16 frames: frame 17 repeats frame 1


def build_cfg(category, subject, art_idx, motion, speed_hz, effect, bg_name, emoji_px, offset, motion_phase, seed, batch,
              accent=None, ui_speed_hz=None, ui_phase=0.0):
    e = CATEGORIES[category][0][subject]
    assert effect in e["effects"], f"{effect} doesn't fit {subject}"
    assert motion in e.get("motions", BASE_MOTIONS), f"{motion} doesn't fit {subject}"
    assert speed_hz in MOTIONS[motion][1], f"{motion} allows speeds {MOTIONS[motion][1]}"
    if category == "ui":
        assert accent in e["accents"] and ui_speed_hz in e["speeds"], f"{subject}: accent {e['accents']}, speed {e['speeds']}"
        art, art_name = f"code:{subject}", f"{accent} {e['name']}"
        who = {"accent": accent, "ui_speed_hz": ui_speed_hz, "ui_phase": ui_phase, "anchor": list(e["anchor"])}
    else:
        art, art_name = e["arts"][art_idx]
        who = {"emotion": subject} if category == "reactions" else {"anchor": list(e["anchor"])}
    return {
        "category": category, "subcategory": subject, **who,
        "art": art, "art_name": art_name, "motion": motion, "speed_hz": speed_hz, "motion_phase": motion_phase,
        "effect": effect, "background": BACKGROUNDS[bg_name], "background_name": bg_name,
        "dark_background": bg_name in ("dark navy", "black"),
        "emoji_px": emoji_px, "offset": offset,
        "frames": FRAMES, "fps": FPS, "size": SIZE,
        "seed": seed, "batch": batch, "generator_version": GENERATOR_VERSION,
    }


def sample_cfg(rng, i, seed, batch, category="reactions", subject=None, effect=None, heldout=None):
    """subject / effect: force them (fill command). heldout: data/splits/heldout.json, so a fill clip never lands on a
    combination reserved for test_unseen."""
    subjects = CATEGORIES[category][0]
    subject = subject or list(subjects)[i % len(subjects)]      # round-robin, so every batch is balanced
    e = subjects[subject]
    motions = e.get("motions", BASE_MOTIONS)
    weights = e.get("motion_weights") or [1 if m != "still" else 0.25 for m in motions]
    held = heldout.get(category, {}) if heldout else {}
    motion = rng.choices(motions, weights=weights)[0]
    effects = e["effects"]                           # "none" is last; give it ~20% of clips
    if effect is None:
        effect = "none" if len(effects) == 1 or rng.random() < 0.2 else rng.choice(effects[:-1])
    tries = 0                                        # same draw order as v0.5, so old batch seeds still reproduce
    while [subject, motion, effect] in held.get("subject_motion_effect", []):   # never a held-out combination
        motion = rng.choices(motions, weights=weights)[0]
        tries += 1
        assert tries < 100, f"every motion is held out for {subject} + {effect}"
    art_idx = rng.randrange(len(e["arts"])) if "arts" in e else 0
    common = (motion, rng.choice(MOTIONS[motion][1]), effect, rng.choice(list(BACKGROUNDS)),
              rng.choice(range(120, 171, 10)), [rng.randint(-14, 14), rng.randint(-10, 10)],
              rng.choice([0.0, 0.25, 0.5, 0.75]), seed, batch)
    if category == "ui":
        bg = common[3]
        accents = [a for a in e["accents"] if a not in bg] or e["accents"]   # no teal on teal, pink on pink
        accents = [a for a in accents if [subject, a] not in held.get("subject_accent", [])] or accents
        return build_cfg(category, subject, art_idx, *common, accent=rng.choice(accents),
                         ui_speed_hz=rng.choice(e["speeds"]), ui_phase=rng.choice(e["start_phases"]))
    return build_cfg(category, subject, art_idx, *common)


def contact_sheets(batch, clips):
    """review/<batch>/: a still sheet (frames 1 and 5 per clip) and an animated grid GIF."""
    out = DATA / "review" / batch
    out.mkdir(parents=True, exist_ok=True)
    cols, cell = 10, 128
    rows = math.ceil(len(clips) / cols)

    grid = []
    for f in range(FRAMES - 1):
        g = Image.new("RGB", (cols * cell, rows * cell), "white")
        for i, (_, frames) in enumerate(clips):
            g.paste(frames[f].resize((cell, cell), Image.LANCZOS), ((i % cols) * cell, (i // cols) * cell))
        grid.append(g)
    grid[0].save(out / "grid.gif", save_all=True, append_images=grid[1:], duration=1000 // FPS, loop=0)

    pair = 96
    sheet = Image.new("RGB", (cols * pair * 2 + cols * 4, rows * (pair + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (cid, frames) in enumerate(clips):
        x, y = (i % cols) * (pair * 2 + 4), (i // cols) * (pair + 14)
        sheet.paste(frames[0].resize((pair, pair)), (x, y + 14))
        sheet.paste(frames[4].resize((pair, pair)), (x + pair, y + 14))
        label = next(cid[len(c) + 1:] for c in CATEGORIES if cid.startswith(c + "-"))
        d.text((x + 2, y + 1), f"{i:02d} {label[:20]}", fill="black")
    sheet.save(out / "sheet.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("--category", default="reactions", choices=CATEGORIES)
    one.add_argument("--subject", default="crying", help="an emotion or object, e.g. crying, fire")
    one.add_argument("--art", type=int, default=0)
    one.add_argument("--motion", default="bounce", choices=MOTIONS)
    one.add_argument("--speed-hz", type=float, default=1.0)
    one.add_argument("--effect", default="tears", choices=EFFECTS)
    one.add_argument("--background", default="light blue", choices=BACKGROUNDS)
    one.add_argument("--accent", choices=ACCENTS, help="UI only")
    one.add_argument("--ui-speed-hz", type=float, default=0.5, help="UI only")
    one.add_argument("--seed", type=int, default=1)
    one.add_argument("--batch", default="b0000")
    bat = sub.add_parser("batch")
    bat.add_argument("--category", default="reactions", choices=CATEGORIES)
    bat.add_argument("--batch", required=True)
    bat.add_argument("--n", type=int, default=100)
    bat.add_argument("--seed", type=int, required=True)
    fil = sub.add_parser("fill", help="exact (category, subject, effect, count) list from a plan file; no review sheets")
    fil.add_argument("--plan", required=True)
    fil.add_argument("--batch", required=True)
    fil.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    if a.cmd == "one":
        cfg = build_cfg(a.category, a.subject, a.art, a.motion, a.speed_hz, a.effect, a.background,
                        150, [0, 0], 0.0, a.seed, a.batch, accent=a.accent, ui_speed_hz=a.ui_speed_hz)
        cid, frames = make_clip(cfg)
        contact_sheets(a.batch + "-" + cid, [(cid, frames)])
        print(cid)
    elif a.cmd == "fill":
        rng = random.Random(a.seed)
        heldout = json.loads((DATA / "splits" / "heldout.json").read_text())
        k = 0
        for category, subject, effect, count in json.loads(Path(a.plan).read_text()):
            for _ in range(count):
                cfg = sample_cfg(rng, k, a.seed * 100000 + k, a.batch, category, subject, effect, heldout)
                cid, _ = make_clip(cfg)
                k += 1
                print(f"{k} {cid}", flush=True)
    else:
        rng = random.Random(a.seed)
        clips = []
        for i in range(a.n):
            cfg = sample_cfg(rng, i, a.seed * 100000 + i, a.batch, a.category)
            clips.append(make_clip(cfg))
            print(f"{i + 1}/{a.n} {clips[-1][0]}", flush=True)
        contact_sheets(a.batch, clips)


if __name__ == "__main__":
    main()
