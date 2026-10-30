# How Bhaav Tiny will be trained, from first principles

This guide assumes you know nothing about the steps. Each part answers three questions:
**what is it, why do we need it, and what would go wrong without it.**

How each fact is known:
- **Verified:** computed or checked by us on our own data
- **Reported:** from Udgam's Drishya Tiny write-up (our private study note; the lab is at https://udgamlabs.com), not re-checked
- **Judgement:** our own reasoning or plan

---

## 0. The whole thing in one minute

We want a program that reads *"a crying emoji bouncing, tears falling, pink background"* and
makes that 2-second clip.

A computer can't "imagine" a video. What it *can* learn is a much simpler skill:
**"here is a slightly noisy picture, tell me which way to nudge it to make it cleaner."**
If it learns that well, we can start from pure static (TV snow) and nudge it, 50 small steps,
until a clean sticker appears. The caption steers every nudge.

Three parts make this possible:

```
                 ┌─────────────────────┐
 caption ──────► │ TEXT ENCODER (T5)   │──► meaning vectors ─────────────┐
                 │ borrowed, frozen    │                                  │
                 └─────────────────────┘                                  ▼
                 ┌─────────────────────┐   small "sketch"   ┌──────────────────────────┐
 video ────────► │ VAE ENCODER         │──► 48×5×16×16 ───► │ GENERATOR (transformer)  │
 (training only) │ borrowed, frozen    │                    │ THE ONLY PART WE TRAIN   │
                 └─────────────────────┘                    └──────────────────────────┘
                 ┌─────────────────────┐                                  │ clean sketch
 sticker  ◄───── │ VAE DECODER         │◄─────────────────────────────────┘
                 │ borrowed, frozen    │
                 └─────────────────────┘
```

**Analogy:** an architect (generator) who only ever draws on tracing paper (the small sketch).
A translator (T5) turns the client's words into notes the architect understands.
A builder (VAE decoder) turns the finished sketch into the real building (the video).
We hire the translator and the builder ready-made; we only train the architect.

---

## 1. Where we are: the raw data (done)

One clip = 17 frames × 256 × 256 pixels × 3 colours (red, green, blue)
= **3,342,336 numbers** (Verified). We have 75,001 clips, each with a caption file.

Splits (done, see `data/README.md`): train 61,569 · val 3,750 · test_seen 3,750 · test_unseen 5,918.

---

## 2. Problem: 3.3 million numbers per clip is too many

**First principle: the cost of a transformer grows with the square of how many pieces it looks at.**
Every piece compares itself with every other piece (that is what "attention" is, see step 5).

If we fed raw pixels, even in 8×8 patches, one clip would be 17 × 32 × 32 = 17,408 pieces.
17,408² ≈ 300 million comparisons per layer, per clip. Too slow and too much memory for one GPU.

But most of those numbers are **redundant**: a yellow face is yellow in thousands of neighbouring
pixels, and frame 5 looks almost exactly like frame 4. The real "information" is far smaller.

**So: compress first, then learn on the compressed version.** That is the job of the VAE.

---

## 3. The VAE: a learned zip file for pictures

### 3a. Autoencoder: the basic idea

An **autoencoder** is two networks back to back:

```
video ──► ENCODER ──► small code ──► DECODER ──► video again
         (squeeze)   (bottleneck)   (unsqueeze)
```

It is trained with one simple rule: **the output must match the input.**
The loss is "how different is the rebuilt video from the original" (e.g. average pixel difference).

**Why does this teach anything useful?** Because of the bottleneck. The code is far too small to
store every pixel, so the encoder is *forced* to keep only what matters (shapes, colours, where
things are, how they move) and the decoder learns to fill in the obvious details (smooth edges,
texture). Nobody tells it what "matters"; it discovers it because that is the only way to rebuild well.

**Analogy:** describing a photo over the phone in 20 words. You'd say "yellow round face, two big
tears, pink background", not the colour of every pixel. The listener redraws it from that.

### 3b. Why "variational"? The problem with a plain autoencoder

A plain autoencoder only cares that *its own* codes can be rebuilt. It may put a crying face at
code [3.1, -870, 0.002…] and a laughing face somewhere completely unrelated, with huge empty gaps
in between. The code space is **messy**: random-looking, any scale, full of holes.

That is fine for zipping, but **bad for generating**, because our generator (step 5) will have to
*produce* codes. If the space is messy, a slightly-off code lands in a hole and decodes to garbage.

A **variational** autoencoder (VAE) fixes this with two changes:
1. The encoder outputs **a small cloud** (a mean and a spread) instead of one exact point, and the
   decoder must rebuild from a random point inside that cloud. So nearby codes must decode to
   similar videos: the space becomes **smooth**.
2. An extra penalty (the **KL term**) pulls all the clouds toward one standard bell curve around 0.
   So codes stay in a **sensible range** with no wild values.

Result: a tidy space where "close codes = similar-looking videos". That is exactly what a generator
needs to work in. (First principle: it's easier to learn to walk on a smooth, bounded map than on a
jagged one with cliffs.)

### 3c. Which VAE we use and how it was made

We use the **Wan2.2 VAE** (Reported: 555M parameters, frozen, the same one Drishya used).
- It was trained by the Wan video team on a very large amount of images and video. Typical
  recipe for VAEs like this (Judgement, from general knowledge, not checked in their paper):
  rebuild loss + KL term + a "perceptual" loss (does it *look* the same to another network) +
  an adversarial loss (a second network tries to spot rebuilt frames, which keeps them sharp).
- It squeezes **4× in time and 16× in each direction of space**, into 48 channels (Verified:
  matches our 17 frames → 5 time steps, 256 px → 16 px).
- It treats the **first frame on its own**, then every 4 frames together. That is *why* our clips
  have 17 = 1 + 4 × 4 frames (Verified by arithmetic).

```
one clip: 17 × 256 × 256 × 3 = 3,342,336 numbers
   VAE encoder ▼
one sketch:  48 × 5 × 16 × 16 =    61,440 numbers   (54× smaller, Verified)
```

### 3d. Why borrow it instead of training our own?

- Training a good video VAE needs millions of clips and far more compute than our whole project.
- It only needs to learn "how pictures look", which is the same for stickers and real video.
- **Frozen** means its weights never change. So we can encode every clip **once**, save the
  sketches to disk (the **latent cache**), and never touch pixels again during training.
  Drishya says this was their single biggest saving (Reported).

### 3e. The risk for *our* data (Judgement)

Wan's VAE learned from real-world video. Stickers are different: flat colours, sharp outlines,
pure white backgrounds. It may blur thin lines (the tick, the Zzz) or shift colours slightly.
**If the VAE can't rebuild our stickers, no generator can fix that**, because the decoder is the
last step. So step 6a below tests this first, before spending on anything else.

---

## 4. The text encoder: turning words into numbers

**First principle: a network can only do arithmetic.** "Crying" means nothing to it; it needs numbers.

A **text encoder** turns a sentence into a list of vectors (one per word-piece) where similar
meanings get similar vectors: "sad" lands near "crying", "LOL" near "laughing".

We borrow **Flan-T5-Base** (Reported: 110M parameters, frozen, max 77 word-pieces), like Drishya.
- It already learned language from billions of sentences. We only have 26,069 different caption
  lines (Verified), mostly from templates: far too little to learn English from scratch.
- Frozen again, so each caption is encoded **once** and cached.

---

## 5. The generator: the only part we train

### 5a. The task: learn to clean noise

Take a clean sketch from our data, call it **x₀**. Take pure random noise, **ε**.
Pick a random "noisiness" **t** between 0 (clean) and 1 (pure noise). Mix them:

```
x_t = (1 − t) · x₀  +  t · ε          ← a point on the straight line from clean to noise
```

Show the generator **x_t**, the **t**, and the **caption**. Ask it: *"which direction points from
clean to noise here?"* The right answer is **v = ε − x₀** (the **velocity**). The loss is the
squared difference between its guess and v.

**Tiny example with single numbers:** clean value x₀ = 3, noise ε = −1, t = 0.25.
x_t = 0.75 × 3 + 0.25 × (−1) = 2.0. Correct velocity v = −1 − 3 = −4.
The model sees "2.0, t = 0.25, caption" and should answer −4. Repeat millions of times on
61,440 numbers at once and it learns what crying emoji sketches look like at every noise level.

This is called **rectified flow** (Reported: Drishya's objective). "Rectified" = the path from noise
to data is a **straight line**, which is simpler to learn and needs fewer steps to follow than the
curvy paths of older diffusion methods.

### 5b. Making a video at the end (generation)

```
start: pure noise (t = 1)
repeat 50 times:  ask the model for v,  step:  x ← x − v × 0.02
end:   a clean sketch (t = 0)  ──►  VAE decoder  ──►  17 frames  ──►  sticker
```

Each step walks a little way back along the straight line towards clean data (Reported: Drishya
uses 50 such "Euler" steps).

### 5c. Why a transformer, and what's inside

The sketch is cut into small blocks (1 time step × 2 × 2 positions), giving **5 × 8 × 8 = 320 pieces**
("tokens", Reported). Each piece is one row of numbers. A **transformer** is a stack of layers
(16 in Drishya, Reported) that repeatedly lets the pieces exchange information:

| Part | What it does | Why we need it |
|---|---|---|
| **Self-attention** | Every piece looks at all 320 pieces | So frame 5's face matches frame 1's face, and the tear lines up with the eye |
| **Cross-attention** | Every piece looks at the caption vectors | So the picture follows the words ("pink background", "tears") |
| **Timestep input** (adaLN) | Tells every layer how noisy the input is | Cleaning heavy noise and light noise are different jobs |
| **3D position code** | Tells each piece *when* and *where* it is | Otherwise it can't tell top-left frame 1 from bottom-right frame 5 |
| **MLP** | A small per-piece calculation after attention | Where most "drawing knowledge" is stored |

This design is called a **DiT** (Diffusion Transformer). Drishya's has 212M trainable numbers
("parameters": the knobs training adjusts) (Reported).

### 5d. Two extra tricks Drishya used (Reported), and why

- **Caption dropout (10%) → classifier-free guidance.** Sometimes train with an empty caption, so
  the model also learns "a generic sticker". At generation, compare "with caption" vs "without" and
  push further in the direction the caption adds (strength 5). Effect: follows the prompt harder.
- **Image-to-sticker in the same model (25% of training).** Replace the noisy first time step with
  the *clean* sketch of the user's image, mark it, and don't score the model on it. The model learns
  "animate this given first frame". That's what our `<id>.png` files are for.

---

## 6. The steps we will actually follow

| # | Step | Where | Why this order |
|---|---|---|---|
| ✅ 1 | Make clips, captions, labels | Mac | Done: 75,001 clips |
| ✅ 2 | Split train / val / test_seen / test_unseen | Mac | Done |
| 3a | **VAE round-trip test:** encode → decode 200 clips, compare to originals by eye and by number | GPU, minutes | If the VAE can't rebuild stickers, stop and rethink before spending more (step 3e) |
| 3b | **Encode all clips once → latent cache** | GPU | About 9.2 GB in bf16 (Verified by arithmetic). Drishya: 475K clips took 6.2 h for $7.40 (Reported), so ours should be well under an hour |
| 3c | **Encode all captions once** with T5 → text cache | GPU, minutes | Same idea: never run T5 during training |
| 4 | **Overfit test:** train on 1 tiny batch until loss ≈ 0 | GPU, minutes | Proves the code can learn at all. If it can't memorise 8 clips, there's a bug |
| 5 | **Short run** on all train data (an hour or two), watch val loss, decode a few samples | GPU | Catches wrong learning rates and broken data before the long run |
| 6 | **Full run** | GPU, hours | The real training. Keep the best checkpoint by val score, not the last one |
| 7 | **Evaluate:** test_seen vs test_unseen | GPU | The gap shows whether it *combines* ideas or memorised them |
| 8 | **Export:** decode → MP4 / GIF / WebP stickers | Mac or GPU | The product |

### What happens in one training step (step 6, repeated many thousands of times)

1. Load a batch of 64 cached sketches + their caption vectors (Reported: Drishya's batch size).
2. For each: pick t, make noise, mix to get x_t. Blank the caption 10% of the time; use the
   clean first frame 25% of the time.
3. The generator guesses the velocity for each.
4. Loss = average squared error between guess and true velocity.
5. **Backpropagation:** work out, for each of the 212M knobs, which way to turn it to reduce the loss.
6. **Optimiser** turns the knobs a little (Reported: Drishya used MuonAdamW, learning rate 2e-4 for AdamW parts).
7. Keep a slow-moving average copy of the knobs (**EMA**, Reported 0.9999). It gives smoother results and is what we release.

### How we will score it

Most video metrics (FVD, CLIP score) were built for real video. **Our advantage:** every clip has an
exact label (`.json`). So we can train a small, separate checker that watches a clip and says
"crying, bounce, tears". Then for each test prompt we generate a sticker and ask the checker:
did it make the right subject? motion? effect? colour? (Judgement: our plan.)
Score that on test_seen and test_unseen; the gap is the headline number.

---

## 7. Our decisions so far, and what is still open (Judgement)

| Decision | Choice for run 1 | Why |
|---|---|---|
| Recipe | Copy Drishya's architecture and settings | Change **one** variable (the data) so results are interpretable |
| VAE / text encoder | Borrow, frozen | Too expensive to train; they encode general looks and language |
| Which VAE | **Wan2.2 VAE** (Apache-2.0), not the MiniMax H3 VAE | H3's licence (checked 2026-10-29) forbids using its outputs to improve any other model, makes anything built on it a "Model Derivative" under its terms, and excludes the US, EU, UK and Korea. Wan2.2 has none of these limits. H3 also has 24 latent channels vs Wan's 48, which would change two things at once vs Drishya |
| Frames | 17 at 8 fps | Matches the VAE's 1 + 4k rule and Drishya |
| Number of steps | **Open.** Drishya: 180,080 steps × 64 = 11.5M clips seen (Reported). On our 61.5K train clips that is ~187 passes; likely too many for a narrow dataset | Decide from step 5: stop when val score stops improving |
| Model size | 212M for run 1; a smaller model is a good second experiment | Our world is narrow (24 emoji images and 10 code-drawn UI elements); it may not need 212M |
| GPU | **Next step to plan.** Drishya used one L40S for 15 h, $13.76 (Reported) | |

---

## 8. Words used in this guide

| Word | Plain meaning |
|---|---|
| Parameter / weight | One adjustable number inside a network. Training = adjusting them |
| Loss | One number saying how wrong the model is. Training pushes it down |
| Latent / sketch | The compressed version of a clip made by the VAE encoder |
| Frozen | Borrowed weights we never change |
| Token / piece | One small block of the input that a transformer handles as a unit |
| Attention | Each piece looking at other pieces and taking in what's relevant |
| Epoch / pass | The model has seen every training clip once |
| bf16 | A shorter number format (16 bits) that halves memory with little accuracy loss |
| Checkpoint | A saved copy of the weights at some point in training |
