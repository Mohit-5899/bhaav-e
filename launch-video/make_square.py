"""Build the 1080x1080 version of the launch film from the same scenes.  (OUR script.)

The scenes size everything from the canvas (container units, and the 3D cameras fit to the short side), so the
square cut only needs a different canvas size. Writes ../launch-video-square; render it there.
Run: python3 make_square.py
"""
import re
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent
DST = SRC.parent / "launch-video-square"
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("renders", ".hyperframes", "node_modules", "make_square.py"))

for f in [DST / "index.html", *sorted((DST / "compositions").glob("*.html"))]:
    s = f.read_text()
    s = s.replace('data-width="1920"', 'data-width="1080"')
    s = s.replace('content="width=1920, height=1080"', 'content="width=1080, height=1080"')
    s = re.sub(r"width: 1920px;", "width: 1080px;", s)
    s = s.replace('data-resolution="landscape"', 'data-resolution="square"')
    f.write_text(s)
print("wrote", DST)
