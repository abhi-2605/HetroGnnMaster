#!/bin/bash
# Rebuild the defence deck: build.js -> editable .pptx -> PDF -> image .pptx (Keynote-safe).
# Needs: node packages pptxgenjs, react, react-dom, react-icons, sharp (point NODE_PATH at them);
#        LibreOffice; a Python with pypdfium2 and python-pptx (PY).
# Usage: NODE_PATH=/path/to/node_modules PY=/path/to/python ./rebuild.sh v6
set -e
V=${1:?version, e.g. v6}
HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
cd "$HERE"
node build.js
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless --convert-to pdf --outdir "$WORK" Thesis_Defence_Menon_editable.pptx >/dev/null 2>&1
cp "$WORK/Thesis_Defence_Menon_editable.pdf" "Thesis_Defence_Menon_$V.pdf"
"${PY:-python3}" - "$HERE" "$V" "$WORK" <<'PY'
import sys, json, pypdfium2 as p
from pptx import Presentation
here, v, work = sys.argv[1:]
d = p.PdfDocument(f"{here}/Thesis_Defence_Menon_{v}.pdf")
for i in range(len(d)):
    d[i].render(scale=3200 / d[i].get_width()).to_pil().save(f"{work}/s{i+1:02d}.png")
pr = Presentation(f"{here}/Thesis_Defence_Menon_editable.pptx")
json.dump([s.notes_slide.notes_text_frame.text if s.has_notes_slide else "" for s in pr.slides], open(f"{work}/notes.json", "w"))
print(len(d), "pages")
PY
node tools/make_image_pptx.js "$WORK" "Thesis_Defence_Menon_$V.pptx"
echo "Now delete the previous version's .pdf/.pptx so only $V remains."
