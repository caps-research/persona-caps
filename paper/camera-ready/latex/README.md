# `paper/camera-ready/latex/` — camera-ready build

> **本目录是 camera-ready 副本，不是送审版。** 送审版在 `paper/latex/`，保持原封不动。
> 工作区总说明、工具链、与送审版的偏离清单见 `../README.md`——先读那份。

Auto-generated from `paper/camera-ready/DRAFT_cr.md` by `md2tex.py`. The markdown
remains the canonical source; LaTeX is regenerated whenever it changes.

送审版的 `.tex` 曾含 65 行「生成之后的手工润色」而 markdown 里没有，导致 regen 会静默还原它们。
这些润色已全部回移到 `md2tex.py` 的 `post_fix()`，`md2tex.py` 现在幂等，**regen 是安全的**。
下面提到 `DRAFT.md` 的地方在本副本里一律指 `DRAFT_cr.md`。

## Files

| File | Source | Purpose |
| --- | --- | --- |
| `main.tex` | hand-written | preamble + author block + `\input` orchestration |
| `main_body.tex` | generated | converted body §1–§7 |
| `appendix_body.tex` | generated | converted appendix sections (Implementation details onwards) |
| `references.bib` | hand-curated | BibTeX entries for every `\cite{...}` in the body |
| `acl.sty` | from `acl-org/acl-style-files` | ACL official style |
| `acl_natbib.bst` | from `acl-org/acl-style-files` | ACL bibliography style |
| `custom.bib` | from `acl-org/acl-style-files` | reserved if extra entries needed |
| `md2tex.py` | hand-written | markdown → LaTeX converter; called by `build.sh` |
| `build.sh` | hand-written | one-command pipeline (regen + mode/author switch + compile + zip) |

## Quick start (one command)

`build.sh` is the single entry point. It regenerates LaTeX from DRAFT.md,
switches between ARR-anonymous / preprint / camera-ready modes, sets the
author block, compiles to PDF when LaTeX is installed, and packages an
Overleaf-uploadable zip.

```bash
cd paper/latex
./build.sh                                            # regen + (compile if pdflatex available)
./build.sh --zip                                      # regen + zip for Overleaf
./build.sh --preprint --author "Yamada Taro" --zip    # de-anonymise for sharing with advisor
./build.sh --review --author "Anonymous ACL submission" --zip   # revert to ARR-anonymous
./build.sh --clean                                    # remove .aux/.bbl/.pdf etc.
./build.sh --help
```

Flags can be combined; the script applies in the order: clean → mode → author → regen → compile → zip.

## Build paths

### Overleaf (recommended, zero install)

```bash
./build.sh --zip
```

Then upload `persona-caps-latex.zip` to overleaf.com (New Project → Upload
Project), set the main file to `main.tex`, hit Recompile.

### Local pdflatex (requires LaTeX install)

```bash
brew install --cask basictex            # one-time, ~100 MB; requires sudo
sudo tlmgr update --self                # one-time
sudo tlmgr install collection-fontsrecommended booktabs hyperref microtype \
                    inconsolata times xcolor

./build.sh                              # auto-detects pdflatex and compiles
open main.pdf
```

### Docker (alternative)

```bash
docker run --rm -v "$PWD:/work" -w /work texlive/texlive \
    sh -c 'pdflatex main && bibtex main && pdflatex main && pdflatex main'
```

## Manually regenerate the converted body only

```bash
python3 md2tex.py   # underlying converter; build.sh calls this
```

## Notes for the camera-ready / submission pass

1. `\usepackage[review]{acl}` is the ARR anonymous-submission setting.
   - Switch to `[final]` for camera-ready.
   - Switch to `[preprint]` for non-anonymous arXiv (adds page numbers).
2. The author block currently reads `Anonymous ACL submission`. **Do not
   de-anonymise** until after acceptance.
3. References are placeholder-quality (arXiv eprints / "others" author
   lists). Replace with conference-proceedings citations during the bib
   pass.
4. Tables are auto-converted with placeholder captions/labels (`TODO
   caption`, `tab:TODO`). Edit each one in `main_body.tex` to give it a
   meaningful caption and a unique `\label{...}` and reference it in the
   text via `Table~\ref{...}`.
5. The status emojis (✅ 🟡 ⬜ 🔧 🎯) and "Working draft" preamble line are
   stripped during conversion; they do not appear in the LaTeX.
6. Search the LaTeX for any remaining `\S<digit>` references — these
   should compile to "§1.2" style cross-refs but for proper ARR style use
   `\Cref{...}` once labels are added.
