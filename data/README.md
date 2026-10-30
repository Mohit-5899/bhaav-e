# Bhaav Tiny data

## Layout

```
data/
  manifest.jsonl     one line per clip: id, path, category, batch, review status
  clips/<category>/<subcategory>/<id>.mp4    the video (what the model learns to make)
                                <id>.txt     the caption (what the user types)
                                <id>.json    the exact label our code used
                                <id>.png     frame 1 (the input for image → sticker)
  review/<batch>/    sheet.png (frames 1 and 5 of every clip) + grid.gif (all clips playing)
  logs/              one log per batch run
  splits/            train / val / test_seen / test_unseen .txt (lists of ids), heldout.json, exclude.txt
  cache/             VAE latents + T5 text embeddings (made later, large, not hand-edited)
```

## Categories

| Folder | What goes in it | Example subfolders |
|---|---|---|
| `reactions/` | Emotion faces | `laugh/`, `crying/`, `heart-eyes/`, `facepalm/`, `angry/` |
| `greetings/` | Occasions and wishes | `good-morning/`, `diwali/`, `birthday/`, `eid/` |
| `text/` | Kinetic text stickers | `haan-bhai/`, `ok-done/`, `chal-nikal/` |
| `animated-emoji/` | Objects and symbols | `fire/`, `heart/`, `party/` |
| `mascot/` | A character we design first | made later |
| `brand/` | Marketing stickers | `sale/`, `new-drop/` |
| `ui/` | Micro-animations | `loader/`, `success-tick/`, `like/` |

Memes made from real film or TV clips are **out of scope**: code can't make them and they are copyrighted.

## One clip = four files with the same id

Id format: `<category>-<subcategory>-<6 digits>`, e.g. `reactions-crying-000001`.

**`<id>.txt`:** the text given to the model. Line 1 is the main caption. Other lines are
other phrasings of the same clip; training picks one at random each time.

```
a yellow crying emoji bouncing, tears falling, light blue background
sad face sticker crying with tears, bouncy
crying emoji, tears, blue background, loops
```

**`<id>.json`:** the exact settings, used to make the captions and to score the model later.

```json
{
  "id": "reactions-crying-000001",
  "category": "reactions",
  "subcategory": "crying",
  "art": "fluent-emoji",
  "motion": "bounce",
  "effect": "tears",
  "background": "#cfe8ff",
  "speed_hz": 1.0,
  "frames": 17,
  "fps": 8,
  "size": 256,
  "seed": 1,
  "batch": "b0000",
  "generator_version": "0.1"
}
```

## Rules

- **Every clip:** 256×256, exactly 17 frames, 8 fps, frame 17 = frame 1.
- **Motion at 2 Hz or slower,** so 8 fps can show it.
- **No text in v1** except in `text/` and `greetings/`, which we start later.
- **Never edit clips by hand.** Fix the generator and re-render, so every clip can be reproduced from its `.json`.
- **Review before scaling:** 1 clip first, then batches of 100, each checked with a contact sheet in `review/`.
- **Test split = unseen combinations** (e.g. crying + spin never in train), to test if the model can combine ideas.

## Splits

Made by `generator/make_splits.py --seed 0` (our code). Re-running gives identical files.

| Split | Clips | Share | What it is for |
|---|---|---|---|
| `train.txt` | 61,569 | 82.1% | Training |
| `val.txt` | 3,750 | 5.0% | Watching training (early stopping, picking a checkpoint). Same combinations as train |
| `test_seen.txt` | 3,750 | 5.0% | Final score on familiar combinations |
| `test_unseen.txt` | 5,918 | 7.9% | Final score on **combinations never in train** |
| `exclude.txt` | 14 | | 13 byte-identical copies + the v0.1 test clip; in no split |

Each category gets the same shares (about 1,250 val, 1,250 test_seen, about 2,000 test_unseen).

**What is held out** (full list in `heldout.json`):
- reactions: 14 subject × motion × effect combinations, e.g. angry + bounce + none, facepalm + wobble + sweat
- animated-emoji: 9 combinations, e.g. clap + pulse + sparkles, hundred + wobble + confetti
- ui: 6 combinations (e.g. success + pulse + sparkles) and 3 colour pairs: blue progress bar, purple spinner, purple toggle

**Rule:** a combination is only held out if each of its pairs (subject+motion, subject+effect, motion+effect)
still appears in train. So test_unseen tests *combining known parts*, never drawing something new.

**How to read the result:** the gap between test_seen and test_unseen is the main number.
A small gap means the model combines ideas; a big gap means it memorised combinations.
