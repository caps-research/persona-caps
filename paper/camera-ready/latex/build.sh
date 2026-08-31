#!/usr/bin/env bash
# build.sh — one-command pipeline to regenerate, compile, and package the
# persona-caps paper.
#
# Usage:
#   ./build.sh                    # regen LaTeX from DRAFT.md, then build PDF if pdflatex available
#   ./build.sh --regen-only       # only regenerate main_body.tex / appendix_body.tex
#   ./build.sh --preprint         # switch acl to [preprint] mode (page numbers, non-anonymous-friendly)
#   ./build.sh --final            # switch acl to [final] mode (camera-ready)
#   ./build.sh --review           # switch acl back to [review] mode (anonymous, ARR default)
#   ./build.sh --author "Your Name <you@uni.edu>"
#   ./build.sh --zip              # produce persona-caps-latex.zip for Overleaf upload
#   ./build.sh --clean            # remove all build artifacts (.aux/.log/.pdf/.bbl/...)
#   ./build.sh --pdf-only         # skip regen, just compile existing .tex
#   ./build.sh --help
#
# Flags can be combined, e.g.:
#   ./build.sh --preprint --author "Yamada Taro" --zip
#
# After --author and --preprint, main.tex is REWRITTEN in place; running
# ./build.sh --review --author "Anonymous ACL submission" reverts to the
# ARR-anonymous default.

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
MAIN="main.tex"

# ---- argument parsing ----
DO_REGEN=1
DO_PDF=1
DO_ZIP=0
DO_CLEAN=0
NEW_MODE=""
NEW_AUTHOR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --regen-only) DO_PDF=0 ;;
    --pdf-only)   DO_REGEN=0 ;;
    --preprint)   NEW_MODE="preprint" ;;
    --final)      NEW_MODE="final" ;;
    --review)     NEW_MODE="review" ;;
    --author)     shift; NEW_AUTHOR="$1" ;;
    --zip)        DO_ZIP=1 ;;
    --clean)      DO_CLEAN=1 ;;
    --help|-h)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift || true
done

# ---- clean ----
if [[ "$DO_CLEAN" -eq 1 ]]; then
  echo "[clean] removing build artefacts"
  rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc main.synctex.gz
  rm -f main_body.aux appendix_body.aux *~
  exit 0
fi

# ---- mode switch ----
if [[ -n "$NEW_MODE" ]]; then
  echo "[mode] switching acl mode to [$NEW_MODE]"
  # Rewrite the \usepackage[<mode>]{acl} line
  python3 - "$MAIN" "$NEW_MODE" <<'PY'
import re, sys
path, mode = sys.argv[1], sys.argv[2]
with open(path) as f:
    text = f.read()
text = re.sub(r"\\usepackage\[\w+\]\{acl\}", rf"\\usepackage[{mode}]{{acl}}", text)
with open(path, "w") as f:
    f.write(text)
print(f"  -> {path} now uses [{mode}]")
PY
fi

# ---- author swap ----
if [[ -n "$NEW_AUTHOR" ]]; then
  echo "[author] setting \\author{$NEW_AUTHOR}"
  python3 - "$MAIN" "$NEW_AUTHOR" <<'PY'
import re, sys
path, author = sys.argv[1], sys.argv[2]
with open(path) as f:
    text = f.read()
# One level of brace nesting has to be allowed: the block being replaced holds a
# \texttt{...} for the email, so "[^{}]*" matched nothing and the swap quietly did
# nothing while still reporting success below.
pattern = re.compile(r"\\author\{(?:[^{}]|\{[^{}]*\})*\}")
# Replace through a function, not a template string: a template reads "\\" as an
# escape and would eat the LaTeX line breaks separating name, affiliation and
# email, and the \And that a second author needs raises "bad escape" outright.
text, n = pattern.subn(lambda m: rf"\author{{{author}}}", text, count=1)
if n != 1:
    sys.exit(f"  !! found no \\author{{...}} block in {path} — nothing was written")
with open(path, "w") as f:
    f.write(text)
print(f"  -> {path} now has \\author{{{author}}}")
PY
fi

# ---- regen ----
if [[ "$DO_REGEN" -eq 1 ]]; then
  echo "[regen] ${PAPER_SRC:=DRAFT_cr.md} -> main_body.tex + appendix_body.tex"
  PAPER_SRC="$PAPER_SRC" python3 md2tex.py
fi

# ---- compile ----
if [[ "$DO_PDF" -eq 1 ]]; then
  # Tectonic first: the submitted PDF was produced by it (Producer: xdvipdfmx),
  # it drives bibtex and the reruns itself, and it needs no system TeX install.
  if command -v tectonic >/dev/null 2>&1; then
    echo "[pdf] compiling with tectonic"
    tectonic -X compile main.tex
  elif command -v pdflatex >/dev/null 2>&1; then
    echo "[pdf] compiling (pdflatex x2 + bibtex + pdflatex)"
    pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
    bibtex main >/dev/null || echo "  (bibtex returned non-zero — likely missing entry, continuing)"
    pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
  else
    echo "[pdf] no LaTeX engine found — skipping local compile."
    echo "       brew install tectonic          (recommended, no sudo)"
    echo "       brew install --cask basictex   (full TeX, needs sudo)"
    echo "       or use Overleaf:  ./build.sh --zip   then upload the zip"
  fi
  if [[ -f main.pdf ]]; then
    echo "[pdf] -> $ROOT/main.pdf  ($(du -h main.pdf | cut -f1), $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2" pages"}'))"
  fi
fi

# ---- zip for Overleaf ----
if [[ "$DO_ZIP" -eq 1 ]]; then
  ZIPNAME="persona-caps-latex.zip"
  echo "[zip] packaging for Overleaf"
  rm -f "$ZIPNAME"
  zip -q "$ZIPNAME" main.tex main_body.tex appendix_body.tex references.bib \
                    acl.sty acl_natbib.bst custom.bib README.md
  echo "  -> $ROOT/$ZIPNAME  ($(du -h "$ZIPNAME" | cut -f1))"
  echo
  echo "  Upload to overleaf.com:"
  echo "    1. New Project -> Upload Project"
  echo "    2. Set main file to main.tex"
  echo "    3. Hit Recompile"
fi

echo "[done]"
