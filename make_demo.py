"""
make_demo.py - copy the Python files and zones.geojson into demo.html.

demo.html runs this repo's Python in the browser (Pyodide). So that it also
works when opened straight from disk, it carries a copy of each file in a
<script type="..." data-file="NAME"> block. Run this after editing any of them:

    python make_demo.py
"""

import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
BLOCK = re.compile(r'(<script type="[^"]+" data-file="([^"]+)">)(.*?)(</script>)', re.S)


def fill(m):
    text = (HERE / m.group(2)).read_text(encoding="utf-8")
    if "</script" in text.lower() or "<!--" in text:
        raise ValueError(f"{m.group(2)} contains text that would break the HTML")
    return f"{m.group(1)}\n{text}{m.group(4)}"


if __name__ == "__main__":
    page = HERE / "demo.html"
    html, n = BLOCK.subn(fill, page.read_text(encoding="utf-8"))
    page.write_text(html, encoding="utf-8")
    print(f"demo.html: refreshed {n} embedded files")
