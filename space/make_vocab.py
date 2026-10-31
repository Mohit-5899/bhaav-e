"""Write space/vocab.json: every subject, motion, effect and background the model was trained on.  (OUR code.)

Read straight from the generator, so the demo's prompt builder can only offer combinations that exist in training,
and builds captions with the same wording as the training captions (generator/bhaav_gen.py, captions()).
Run: .venv/bin/python space/make_vocab.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generator"))
import bhaav_gen as g  # noqa: E402

LABELS = {"reactions": "Emotion faces", "animated-emoji": "Animated emoji", "ui": "UI animations"}

subjects = []
for cat, (table, _) in g.CATEGORIES.items():
    for key, e in table.items():
        if cat == "ui":
            arts = [f"{a} {e['name']}" for a in e["accents"]]
        else:
            arts = [name for _, name in e["arts"]]
        subjects.append(dict(
            category=cat, group=LABELS[cat], key=key, arts=arts,
            action=e.get("action"),                                   # UI only: the element's own animation
            motions=[[m, g.MOTIONS[m][2]] for m in e.get("motions", g.BASE_MOTIONS)],
            effects=[[x, g.EFFECTS[x][1]] for x in e["effects"]],
        ))

vocab = dict(subjects=subjects, backgrounds=list(g.BACKGROUNDS))
out = Path(__file__).resolve().parent / "vocab.json"
out.write_text(json.dumps(vocab, indent=1, ensure_ascii=False) + "\n")
print(out, len(subjects), "subjects,", len(vocab["backgrounds"]), "backgrounds")
