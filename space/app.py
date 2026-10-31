"""Bhaav-E demo: type or build a prompt, get a 2-second looping sticker.  (OUR code.)"""
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import torch

from bhaav import Bhaav, save_gif

HERE = Path(__file__).parent
VOCAB = json.loads((HERE / "vocab.json").read_text())
SUBJECTS = {s["key"]: s for s in VOCAB["subjects"]}
GROUPS = ["Emotion faces", "Animated emoji", "UI animations"]
SPEEDS = {"Fast · 20 steps": 20, "Balanced · 30 steps": 30, "Best · 50 steps": 50}
NAMES = {"heart-eyes": "Heart eyes", "dots-loader": "Dots loader", "thumbs-up": "Thumbs up", "hundred": "100",
         "idea": "Light bulb", "party": "Party popper", "clap": "Clapping hands"}
EXAMPLES = json.loads((HERE / "examples" / "examples.json").read_text())
MAX_SEED = 2**31 - 1

torch.set_num_threads(os.cpu_count() or 2)
PIPE = Bhaav()
ON_GPU = PIPE.dev == "cuda"
CACHE = {}                                  # (prompt, steps, cfg, seed) -> gif path; repeats are instant
OUT = Path(tempfile.mkdtemp(prefix="bhaav_"))


def name(key):
    return NAMES.get(key, key.replace("-", " ").capitalize())


def subjects_in(group):
    return [(name(s["key"]), s["key"]) for s in VOCAB["subjects"] if s["group"] == group]


def caption(subject, look, motion, effect, background):
    """Same wording as the training captions (generator/bhaav_gen.py, captions(), line 1)."""
    s = SUBJECTS[subject]
    verb = dict(s["motions"])[motion]
    if s["action"]:                                       # UI: the element's own animation comes first
        verb = s["action"] + ("" if motion == "still" else f", {verb}")
    phrase = dict(s["effects"])[effect]
    article = "an" if look[0] in "aeiou" else "a"
    return f"{article} {look} {verb}{', ' + phrase if phrase else ''}, {background} background"


def on_group(group):
    first = subjects_in(group)[0][1]
    return (gr.update(choices=subjects_in(group), value=first), *on_subject(first))


def on_subject(subject):
    s = SUBJECTS[subject]
    motions = [(verb if m != "still" or not s["action"] else "just its own animation", m) for m, verb in s["motions"]]
    effects = [("no effect" if e == "none" else p, e) for e, p in s["effects"]]
    default_motion = "still" if s["action"] else s["motions"][0][0]
    default_effect = s["effects"][0][0]
    return (gr.update(choices=s["arts"], value=s["arts"][0]), gr.update(choices=motions, value=default_motion),
            gr.update(choices=effects, value=default_effect))


def build(subject, look, motion, effect, background):
    if not (subject and look and motion and effect and background):
        return gr.update()
    if look not in SUBJECTS[subject]["arts"] or motion not in dict(SUBJECTS[subject]["motions"]) \
            or effect not in dict(SUBJECTS[subject]["effects"]):
        return gr.update()                                # mid-update after a subject change; the next event fixes it
    return caption(subject, look, motion, effect, background)


def pick_example(evt: gr.SelectData):
    return EXAMPLES[evt.index]["prompt"]


def generate(prompt, speed, cfg, seed, randomize, progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Write a prompt or use the prompt builder first.")
    seed = random.randint(0, MAX_SEED) if randomize else int(seed)
    steps = SPEEDS[speed]
    key = (prompt, steps, float(cfg), seed)
    if key in CACHE:
        return CACHE[key], seed, f"Seed {seed} · {steps} steps · from cache"
    t0 = time.time()
    progress(0, desc="Reading the prompt")
    frames = PIPE.generate(prompt, steps=steps, cfg=float(cfg), seed=seed,
                           on_step=lambda i, n: progress(i / (n + 1), desc=f"Denoising step {i}/{n}"))
    progress(steps / (steps + 1), desc="Decoding frames")
    path = save_gif(frames, OUT / f"bhaav_{abs(hash(key))}.gif")
    CACHE[key] = str(path)
    return str(path), seed, f"Seed {seed} · {steps} steps · guidance {cfg:g} · {time.time() - t0:.0f} s on {'GPU' if ON_GPU else 'CPU'}"


CSS = """
.gradio-container {max-width: 1120px !important; margin: auto}
#hero h1 {font-size: 2.3rem; margin-bottom: 0.2rem}
#hero p {margin: 0.15rem 0; opacity: 0.85}
#hero .links a {margin-right: 0.9rem; font-weight: 600}
#output img {border-radius: 14px; image-rendering: auto}
#gen {min-height: 48px; font-size: 1.05rem}
.note {font-size: 0.85rem; opacity: 0.8}
"""

THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"])

speed_default = "Best · 50 steps"
eta = "about 15–40 s on a GPU" if ON_GPU else "about 1–3 minutes on CPU; most of it is decoding the frames"

with gr.Blocks(title="Bhaav-E · animated sticker generator") as demo:
    gr.HTML(f"""
    <div id="hero">
      <h1>Bhaav-E 🎞️</h1>
      <p><b>Type a prompt, get a 2-second looping sticker.</b> A 212M-parameter text-to-video model trained from
         scratch for about $12 on 75K code-generated stickers.</p>
      <p class="links"><a href="https://huggingface.co/Mohit5899/bhaav-e" target="_blank">Model</a>
         <a href="https://github.com/Mohit-5899/bhaav-e" target="_blank">Code &amp; results</a></p>
    </div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Accordion("Prompt builder: pick from what the model knows", open=True):
                group = gr.Radio(GROUPS, value=GROUPS[0], label="Category")
                with gr.Row():
                    subject = gr.Dropdown(subjects_in(GROUPS[0]), value="crying", label="Subject")
                    look = gr.Dropdown(SUBJECTS["crying"]["arts"], value="crying emoji", label="Look")
                with gr.Row():
                    motion = gr.Dropdown(label="Motion")
                    effect = gr.Dropdown(label="Effect")
                background = gr.Dropdown(VOCAB["backgrounds"], value="light blue", label="Background")
            prompt = gr.Textbox(label="Prompt", lines=2, max_lines=4,
                                value="a crying emoji bouncing, tears falling, light blue background",
                                info="Filled in by the builder. You can edit it; prompts in the same style work best.")
            speed = gr.Radio(list(SPEEDS), value=speed_default, label="Quality",
                             info="More steps = cleaner small effects. On CPU the steps are cheap; decoding the video takes most of the time.")
            with gr.Accordion("Advanced", open=False):
                cfg = gr.Slider(1.0, 8.0, value=3.0, step=0.5, label="Guidance",
                                info="How strongly to follow the prompt. 3 tested best; high values can add unasked effects.")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, minimum=0, maximum=MAX_SEED, label="Seed")
                    randomize = gr.Checkbox(value=False, label="Random seed each time")
            run = gr.Button("Generate sticker", variant="primary", elem_id="gen")
            gr.Markdown(f"<span class='note'>⏱️ {eta}. One sticker at a time; you are queued if someone else is "
                        f"generating.</span>")
        with gr.Column(scale=4):
            out = gr.Image(label="Your sticker (GIF)", type="filepath", elem_id="output", height=360,
                           interactive=False)
            info = gr.Markdown("")
            gr.Markdown("### Examples\nGenerated by this model, seed 0, first try. Click one to use its prompt.")
            gallery = gr.Gallery([(str(HERE / "examples" / e["file"]), e["label"]) for e in EXAMPLES],
                                 columns=4, height="auto", allow_preview=False, show_label=False)

    with gr.Accordion("About the model and its limits", open=False):
        gr.Markdown("""
**How it works.** A frozen Flan-T5 encoder turns the prompt into vectors. Our VideoDiT (a diffusion transformer,
211.92M parameters, trained with rectified flow) starts from noise and cleans it up step by step, guided by the
prompt, in the compressed space of the frozen Wan2.2 video VAE. The VAE then decodes 17 frames of 256 × 256 (8 fps).

**What it knows.** 30 subjects (10 emotion faces, 10 animated emoji, 10 UI animations), 9 motions, 12 effects,
12 background colours. It cannot draw subjects outside this set. The builder only offers combinations it was
trained on, but it can also handle new mixes: on held-out combinations it gets 3.3% wrong, against 2.0% on seen ones.

**Known weakness.** It sometimes adds the effect a subject usually has (a laughing face with tears of joy) when
none was asked for.

Wan2.2 VAE and Flan-T5 are Apache-2.0; emoji art in the training data is Microsoft Fluent Emoji (MIT).
""")

    group.change(on_group, group, [subject, look, motion, effect])
    subject.change(on_subject, subject, [look, motion, effect])
    for c in (subject, look, motion, effect, background):
        c.change(build, [subject, look, motion, effect, background], prompt)
    gallery.select(pick_example, None, prompt)
    run.click(generate, [prompt, speed, cfg, seed, randomize], [out, seed, info])
    prompt.submit(generate, [prompt, speed, cfg, seed, randomize], [out, seed, info])
    demo.load(lambda: on_subject("crying"), None, [look, motion, effect])

if __name__ == "__main__":
    # --share: a temporary public gradio.live link, used when running from the Colab notebook
    demo.queue(default_concurrency_limit=1, max_size=20).launch(css=CSS, theme=THEME, share="--share" in sys.argv)
