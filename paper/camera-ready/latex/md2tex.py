#!/usr/bin/env python3
"""
Convert paper/DRAFT.md to LaTeX body (sections only — no preamble/title).
Outputs to paper/latex/body.tex.

Conversions handled:
  # / ## / ### → \section / \subsection / \subsubsection
  **bold**     → \textbf{bold}
  *italic*     → \textit{italic} (only outside math)
  `code`       → \texttt{code} (escaped)
  Markdown table → LaTeX tabular (best-effort)
  Bulleted list (- ...) → itemize
  Numbered list (1. ...) → enumerate
  ```bash ... ``` → verbatim
  ✅ / 🟡 / ⬜ status emojis stripped
  $...$ math passthrough
  § references stay as \S
"""
from __future__ import annotations
import re
import sys
from collections import Counter
from pathlib import Path

import os as _os
SRC = Path(__file__).parent.parent / _os.environ.get("PAPER_SRC", "DRAFT.md")
MAIN_DST = Path(__file__).parent / "main_body.tex"
APPENDIX_DST = Path(__file__).parent / "appendix_body.tex"
POST_REFS_DST = Path(__file__).parent / "post_refs.tex"
# Marker line in body where appendix begins (first \section after the conclusion).
APPENDIX_TITLE_KEYWORD = "Implementation details"
# Section titles that EMNLP/ARR convention puts AFTER \bibliography as
# unnumbered \section*{...} (they don't count against the 8-page main-body
# budget). We extract them out of the main body and emit them to post_refs.tex.
# Order here is the order they are emitted, and ACLPUB puts acknowledgements
# last of the three, immediately before the references.
POST_REFS_SECTIONS = ["Limitations", "Ethics and Responsible Use", "Acknowledgements"]
# ACL requires these to be unnumbered, so pointers at them cannot print a
# section number and are hyperlinked by name instead.
UNNUMBERED_LABELS = {"Limitations": "sec:limitations",
                     "Ethics and Responsible Use": "sec:ethics",
                     "Acknowledgements": "sec:acks"}

# ---------------------------------------------------------------------------
# Cross-references
#
# The draft writes pointers as literal text ("App. A.9.6", "§4.4"). Inserting an
# appendix subsection renumbers everything after it, which silently redirects
# every literal pointer while still compiling cleanly. Pointers therefore have to
# resolve through the numbering rather than restate it.
#
# Two kinds of target exist. A number on a real heading is owned by LaTeX, so it
# becomes a \ref{} and follows renumbering on its own. A number on a bold run-in
# label ("**A.9.6 — ...**") is hand-written and cannot renumber itself, so it only
# gets a hyperlink and keeps its literal text; those are safe as long as no real
# heading is inserted above the one they hang off.
# ---------------------------------------------------------------------------
_REF_TARGETS: dict[str, str] = {}       # "4.5" -> "sec:4.5" (real heading)
_PSEUDO_TARGETS: dict[str, str] = {}    # "A.9.6" -> "app:A.9.6" (run-in label)
_UNNUMBERED_NUMS: dict[str, str] = {}   # "6" -> "sec:limitations" (number is dropped)
_REF_STATS: Counter = Counter()
_REF_UNRESOLVED: Counter = Counter()
# Run-in labels whose own number was deliberately left literal, so the audit at the
# end can tell them apart from pointers that simply failed to convert -- a warning
# that can never reach zero is one people learn to skip past.
_REF_ANCHORED: Counter = Counter()

# A heading number, either appendix-style (A.9.6) or main-body style (4.5).
_NUM = r"[A-Z]\.\d+(?:\.\d+)?|\d+(?:\.\d+){1,2}"
_HEADING_NUM = re.compile(
    rf"^#{{2,4}}\s+(?:Appendix\s+([A-Z])\s*[—-]|({_NUM})\s|(\d+)\s)")
# Run-in labels are bold, start the line, and lead with a section number. The
# number needs at least one dot so that bold text opening with a bare figure
# ("**4 / 10 conditions**") is not mistaken for a label.
_RUNIN_NUM = re.compile(rf"^\*\*§?({_NUM})[ .]")
# The separator is captured, not normalised: the draft writes both "App. C" and
# "App.~A.9.5", and turning a space into a non-breaking one moves the line breaks.
# Two spellings of the section sign have to reach this pattern. The draft mixes the
# unicode "§" with the literal "\S~" it renders to, and only the first was anchored
# here, so a pointer written the second way passed through as plain text -- including
# the abstract's and six in the contribution list.
# The appendix branch needs its own alternative for a related reason: "[A-Z]?\d+"
# wants a digit immediately after the optional letter, so "§A.1" never matched and
# an appendix pointer was only ever converted when it was spelled "App.~A.1". That
# is the case R2 named as the laborious one.
_POINTER = re.compile(
    r"(?P<app>App\.|Appendix)(?P<asep>[ ~])(?P<anum>[A-Z](?:\.\d+)*)"
    r"|§(?P<name>Limitations|Ethics)"
    r"|(?:§|\\S~)(?P<ssep>[ ~]?)(?P<snum>[A-Z](?:\.\d+)+|[A-Z]?\d+(?:\.\d+)*)")


def _label_key(num: str) -> str:
    return ("app:" if num[0].isalpha() else "sec:") + num


def scan_targets(src: list[str]) -> None:
    """Record every section number in the draft and how it is defined."""
    _REF_TARGETS.clear()
    _PSEUDO_TARGETS.clear()
    _UNNUMBERED_NUMS.clear()
    runins: list[str] = []
    in_fence = False
    for line in src:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if m := _HEADING_NUM.match(line):
            num = m.group(1) or m.group(2) or m.group(3)
            title = re.sub(r"^#+\s+\S+\s+", "", line).strip()
            if title in UNNUMBERED_LABELS:
                _UNNUMBERED_NUMS[num] = UNNUMBERED_LABELS[title]
            else:
                _REF_TARGETS[num] = _label_key(num)
        elif m := _RUNIN_NUM.match(line):
            runins.append(m.group(1))
    # A bold line opening with a number that a real heading already owns is a
    # pointer, not a label: "**§4.5 / §4.5.2 5-vendor matrix.**" in Appendix A.7
    # describes §4.5, it does not define it. Labelling it would put a second
    # \label{sec:4.5} in the appendix and \ref would resolve to whichever came last.
    for num in runins:
        if num not in _REF_TARGETS and num not in _UNNUMBERED_NUMS:
            _PSEUDO_TARGETS.setdefault(num, _label_key(num))


def convert_refs(s: str, self_num=None) -> str:
    """Turn literal section pointers into \\ref{} or, for run-in labels, a hyperlink.

    self_num is the number the line being converted *defines* (a run-in label). Its
    own number has to stay literal or the paragraph becomes a link to itself; every
    other pointer on the line still converts, so "**A.9.5 -- §4.5 cross-vendor full
    grid**" keeps §4.5 live while the A.9.5 that names the paragraph does not move.
    """
    def repl(m):
        if self_num is not None and (m.group("anum") or m.group("snum")) == self_num:
            _REF_ANCHORED[self_num] += 1
            return m.group(0)
        if m.group("app"):
            num, lead = m.group("anum"), m.group("app") + m.group("asep")
            if key := _REF_TARGETS.get(num):
                _REF_STATS["ref"] += 1
                return f"{lead}\\ref{{{key}}}"
            if key := _PSEUDO_TARGETS.get(num):
                _REF_STATS["hyperref"] += 1
                return f"{lead}\\hyperref[{key}]{{{num}}}"
            _REF_UNRESOLVED[f"App. {num}"] += 1
            return m.group(0)
        if name := m.group("name"):
            _REF_STATS["hyperref"] += 1
            label = UNNUMBERED_LABELS["Limitations" if name == "Limitations"
                                      else "Ethics and Responsible Use"]
            return f"\\hyperref[{label}]{{{name}}}"
        num, lead = m.group("snum"), "\u00a7" + m.group("ssep")
        if num in _UNNUMBERED_NUMS:
            # ACL prints no number for these, so "§6" points at nothing a reader
            # can find. Flagged rather than rewritten: the surrounding sentence
            # has to name the section instead, which only reads well by hand.
            _REF_UNRESOLVED[f"\u00a7{num} (unnumbered — name it instead)"] += 1
            return m.group(0)
        if key := _REF_TARGETS.get(num):
            _REF_STATS["ref"] += 1
            # § is still literal here; the caller rewrites it to \S~ afterwards.
            return f"{lead}\\ref{{{key}}}"
        if key := _PSEUDO_TARGETS.get(num):
            _REF_STATS["hyperref"] += 1
            return f"\\hyperref[{key}]{{{lead}{num}}}"
        _REF_UNRESOLVED[f"\u00a7{num}"] += 1
        return m.group(0)
    return _POINTER.sub(repl, s)


_LITERAL_POINTER = re.compile(r"(?:\\S~|App\.~)([A-Z](?:\.\d+)*|\d+(?:\.\d+)*)")
# One level of nesting is enough for the titles here, e.g. the "$P^{\!2}$" in A.4.1.
_HEADING_ARG = re.compile(r"\\(?:sub){0,2}section\*?\{((?:[^{}]|\{[^{}]*\})*)\}")


def audit_literal_pointers(*parts):
    """Find pointer-shaped text that is still literal in the generated .tex.

    _REF_UNRESOLVED can only report strings the pattern matched and then failed to
    resolve, so a pointer spelled in a way the pattern did not recognise was invisible
    to it -- which is how 33 of them survived runs that reported nothing wrong, with
    every label they needed already emitted. Auditing the output instead needs no
    foresight about spellings: by this stage every pointer reads "\\S~<num>" or
    "App.~<num>". Returns (misses, in_titles).
    """
    misses: Counter = Counter()
    in_titles: Counter = Counter()

    def take_title(m):
        for num in _LITERAL_POINTER.findall(m.group(1)):
            in_titles[num] += 1
        return ""

    for part in parts:
        # A converted pointer still reads "\S~<num>" inside the braces of
        # \hyperref[key]{\S~<num>}, so scanning raw text counts live links as misses.
        part = re.sub(r"\\hyperref\[[^\]]*\]\{[^}]*\}", "", part)
        part = re.sub(r"\\ref\{[^}]*\}", "", part)
        # Headings keep their pointers literal on purpose (see convert_inline), and
        # only what survives the number-prefix stripping counts, which is why this
        # reads the emitted title rather than tallying during conversion.
        part = _HEADING_ARG.sub(take_title, part)
        for num in _LITERAL_POINTER.findall(part):
            misses[num] += 1
    return misses - _REF_ANCHORED, in_titles


def escape_text(s: str) -> str:
    """Escape LaTeX special chars in plain text (NOT inside math/code/verbatim)."""
    # Keep math intact
    parts = re.split(r"(\$[^$]*\$)", s)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
            continue
        # escape & % # _ ^ ~ < > and stray { }
        p = p.replace("\\", r"\textbackslash{}")
        p = p.replace("&", r"\&")
        p = p.replace("%", r"\%")
        p = p.replace("#", r"\#")
        p = p.replace("_", r"\_")
        p = p.replace("^", r"\textasciicircum{}")
        p = p.replace("~", r"\textasciitilde{}")
        # Keep < > as text
        p = p.replace("<", r"\textless{}").replace(">", r"\textgreater{}")
        out.append(p)
    return "".join(out)


def _break_long_math(seg: str) -> str:
    """Let TeX break a long comma-separated tuple in inline math.

    Within math, TeX only breaks at relations and binary operators, so a tuple
    like a = (situation, behavior, [affect], ...) is a single unbreakable box. At
    57 visible characters against a ~218pt column that box claims a line of its
    own and the lines above stretch to compensate -- measured at 4.9x the body
    word spacing. Commas are the natural break points. Commas nested in braces
    (subscripts, \\text arguments) are left alone.
    """
    if seg.startswith("$$") or seg.count(",") < 3:
        return seg
    visible = re.sub(r"\\text\{([^}]*)\}", r"\1", seg)
    visible = re.sub(r"\\[a-zA-Z]+", " ", visible)
    visible = re.sub(r"[{}$\\]", "", visible)
    if len(re.sub(r"\s+", " ", visible).strip()) < 32:
        return seg
    out: list[str] = []
    depth = i = 0
    while i < len(seg):
        ch = seg[i]
        if ch == "\\" and i + 1 < len(seg):      # \{ and \} are glyphs, not groups
            out.append(seg[i:i + 2])
            i += 2
            continue
        out.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(r"\allowbreak ")
        i += 1
    return "".join(out)


def convert_inline(s: str, refs: bool = True, self_num=None) -> str:
    """Run inline conversions on a single line of markdown after the math is preserved.

    Pass refs=False for section titles: a \\ref{} inside \\section{} would end up in
    the PDF bookmark string, and the "Appendix A — " lead-in of an appendix heading
    otherwise reads as a pointer at itself. A run-in label needs the narrower
    self_num instead, which spares only its own number (see convert_refs).
    """
    # Preserve math first
    math_segments: list[str] = []
    def stash_math(m):
        math_segments.append(_break_long_math(m.group(0)))
        return f"@@MATH{len(math_segments)-1}@@"
    # A '$' that is preceded by '\' (i.e. the markdown-escaped literal "\$") must
    # not open a math segment, otherwise text like "**USD \$30 on X**, **USD \$60..."
    # gets greedily stashed and the surrounding bold markers vanish.
    # Display math '$$...$$' must be stashed FIRST, otherwise the single-$ pass
    # below splits it into '$$'+contents+'$$' segments and escapes underscores in
    # the contents.
    s = re.sub(r"(?<!\\)\$\$[^$]+?\$\$", stash_math, s)
    s = re.sub(r"(?<!\\)\$[^$]*?(?<!\\)\$", stash_math, s)

    url_segments: list[str] = []
    def stash_url(m):
        url_segments.append(m.group(0))
        return f"@@URL{len(url_segments)-1}@@"
    s = re.sub(r"\\url\{[^}]+\}", stash_url, s)

    # Preserve inline code first (escape contents)
    code_segments: list[str] = []
    INLINE_CODE_UNICODE_FIXES = {
        "−": "-",
        "→": "->",
        "≥": ">=",
        "≤": "<=",
        "≈": "~=",
        "±": "+/-",
        "—": "--",
        "–": "-",
        "…": "...",
        "≡": "==",
        "τ": "tau",
        "π": "pi",
        "σ": "sigma",
        "ρ": "rho",
        "Δ": "Delta",
        "θ": "theta",
        "φ": "phi",
        "✓": "[v]",
        "✗": "[x]",
        "↔": "<->",
    }
    # TeX never hyphenates \texttt{}, so a long identifier is a solid block: it
    # cannot end a line, and because acl.sty runs \sloppy the roman glue on the
    # line before it stretches instead. That, not margin overflow, is what opens
    # the wide gaps around paths. \allowbreak sets no glyph, so a break leaves
    # the identifier literal -- no hyphen is invented that a reader could mistake
    # for part of a path.
    # Marked on the raw text and only turned into \allowbreak after escaping, so
    # a rule can never match inside a control sequence such as \textbackslash
    # that escaping is about to introduce.
    BRK = "\x00"
    _SEP_BREAK = re.compile(r"(==|[/_,-])")
    # A dot qualifies only when letters follow it *and* something non-letter
    # follows those, i.e. it separates two parts of a name (IfThenAtom.render)
    # rather than introducing a trailing extension. Breaking at ".py" or ".json"
    # strands two characters on the next line, and a dot before digits would cut
    # 1.26.4 mid-version.
    _DOT_BREAK = re.compile(r"\.(?=[A-Za-z]+[^A-Za-z])")

    def _mark_breaks(body: str) -> str:
        body = _DOT_BREAK.sub("." + BRK, body)
        return _SEP_BREAK.sub(r"\1" + BRK, body)

    def stash_code(m):
        body = _mark_breaks(m.group(1))
        # Escape special chars inside \texttt{}
        body = body.replace("\\", r"\textbackslash{}")
        body = body.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
        body = body.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
        body = body.replace("^", r"\textasciicircum{}").replace("~", r"\textasciitilde{}")
        for k, v in INLINE_CODE_UNICODE_FIXES.items():
            body = body.replace(k, v)
        body = body.replace(BRK, r"\allowbreak{}")
        code_segments.append(rf"\texttt{{{body}}}")
        return f"@@CODE{len(code_segments)-1}@@"
    s = re.sub(r"`([^`]+)`", stash_code, s)

    # Bold (non-greedy so bold spans containing inner *italic* still match)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    # Italic (single * not at line start to avoid list bullets — but we already strip leading - first)
    s = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"\\textit{\1}", s)

    # Resolve pointers before § is rewritten, so the patterns can still see it.
    if refs:
        s = convert_refs(s, self_num)
    # § character → \S~ (non-breaking space avoids "\S A" being parsed as command \SA)
    s = s.replace("§", r"\S~")
    # Various Unicode → LaTeX-safe substitutions (pdflatex-compatible).
    UNICODE_FIXES = {
        "—": "---",        # em-dash
        "–": "--",          # en-dash
        "−": r"$-$",       # minus sign
        "±": r"$\pm$",
        "→": r"$\to$",
        "⇒": r"$\Rightarrow$",
        "∈": r"$\in$",
        "Δ": r"$\Delta$",
        "²": r"$^{2}$",
        "·": r"$\cdot$",
        "…": r"\ldots{}",
        "✓": r"\ding{51}",   # via pifont in main.tex
        "✗": r"\ding{55}",
        "↔": r"$\leftrightarrow$",
        "⟂": r"$\perp$",
        "≡": r"$\equiv$",
        "τ": r"$\tau$",
        "π": r"$\pi$",
        "σ": r"$\sigma$",
        "ρ": r"$\rho$",
        "θ": r"$\theta$",
        "φ": r"$\phi$",
    }
    # Apply outside math segments only — math segments are stashed already
    parts2 = re.split(r"(@@(?:MATH|CODE|URL)\d+@@)", s)
    for i, p in enumerate(parts2):
        if p.startswith("@@") and p.endswith("@@"):
            continue
        for k, v in UNICODE_FIXES.items():
            p = p.replace(k, v)
        parts2[i] = p
    s = "".join(parts2)
    # Smart-quote markdown style "..." → ``...''
    # Be conservative: only convert paired straight ASCII quotes
    def _quotes(text: str) -> str:
        out = []
        toggle = False
        for ch in text:
            if ch == '"':
                out.append("``" if not toggle else "''")
                toggle = not toggle
            else:
                out.append(ch)
        return "".join(out)
    s = _quotes(s)
    # Multiplication × outside math → $\times$
    s = re.sub(r"(\d)\s*×\s*(\d)", r"\1$\\times$\2", s)
    s = s.replace("×", r"$\times$")
    # Approx ≈ outside math
    s = s.replace("≈", r"$\approx$")
    # ≥ ≤
    s = s.replace("≥", r"$\geq$").replace("≤", r"$\leq$")
    # En-dash / em-dash already work in Latin-1; nothing to do.

    # Cross-section refs like "§4.1" already became "\S4.1" — fine
    # Markdown links [text](url) → \href{url}{text}
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\\href{\2}{\1}", s)

    # Now escape remaining special chars in plain text portions (excluding stashed segments)
    # We need to escape but keep our @@MATH@@ / @@CODE@@ markers
    parts = re.split(r"(@@(?:MATH|CODE|URL)\d+@@)", s)
    for i, p in enumerate(parts):
        if p.startswith("@@") and p.endswith("@@"):
            continue
        # escape & % _ in plain-text portions. Math/code segments are already stashed
        # and unaffected. Underscores inside \texttt{}/\textbf{}/\textit{}/\href{} that
        # were inserted earlier in this function get escaped here as well, which is
        # correct for pdflatex.
        # Don't double-escape: only escape chars that are NOT already preceded by '\'
        p = re.sub(r"(?<!\\)&", r"\\&", p)
        p = re.sub(r"(?<!\\)%", r"\\%", p)
        p = re.sub(r"(?<!\\)_", r"\\_", p)
        # # at start of line is handled at block level; here in inline keep as-is
        parts[i] = p
    s = "".join(parts)

    # Restore stashed segments
    for i, seg in enumerate(math_segments):
        s = s.replace(f"@@MATH{i}@@", seg)
    for i, seg in enumerate(code_segments):
        s = s.replace(f"@@CODE{i}@@", seg)
    for i, seg in enumerate(url_segments):
        s = s.replace(f"@@URL{i}@@", seg)

    # Strip status emojis — be specific
    for emoji in ["✅", "🟡", "⬜", "🔧", "🎯"]:
        s = s.replace(emoji, "")
    # tidy any " (null result, reported)" style suffix attached to headings
    s = re.sub(r"\s+$", "", s)
    return s


_TABLE_COUNTER = [0]
# Per-table caption + label populated by a markdown HTML-comment marker
# emitted IMMEDIATELY before the table:
#   <!-- caption: ...the real caption sentence...  label: tab:foo -->
# The convert() loop picks the marker up, stashes it in this single-slot
# buffer, and convert_table() consumes it for the next table.
_PENDING_CAPTION: list[str] = [""]
_PENDING_LABEL:   list[str] = [""]
_PENDING_WIDE:    list[bool] = [False]


def convert_table(lines: list[str]) -> list[str]:
    """Convert a contiguous block of markdown table lines into LaTeX tabular.

    If a <!-- caption: ... [label: tab:foo] --> marker was seen immediately
    before this table, use it. Otherwise fall back to a neutral generic
    caption (still preferable to a "TODO" placeholder in a reviewer-facing
    PDF, and the markdown text surrounding the table always carries the full
    natural-language description anyway).
    """
    # Find header (first line), separator (second), body rows
    header = lines[0]
    cols = [c.strip() for c in header.strip().strip("|").split("|")]
    sep = lines[1].strip().strip("|").split("|")
    aligns = []
    for s in sep:
        s = s.strip()
        if s.startswith(":") and s.endswith(":"):
            aligns.append("c")
        elif s.endswith(":"):
            aligns.append("r")
        elif s.startswith(":"):
            aligns.append("l")
        else:
            aligns.append("l")
    body = lines[2:]
    _TABLE_COUNTER[0] += 1
    n = _TABLE_COUNTER[0]
    out = []
    # Decide rendering strategy by max row width (ACL single column fits ~40
    # chars at \small, the full text width ~83):
    #   max_w <= 40 : plain `table` + `tabular` at \small
    #   max_w <= 55 : plain `table` + \fitwidth{\columnwidth}{tabular}
    #   max_w >  55 : `table*` + \fitwidth{\textwidth}{tabular}
    # Pure column count is too blunt; this width estimate matches ACL's column
    # geometry and is what real EMNLP papers use.
    body_rows = [
        [c.strip() for c in r.strip().strip("|").split("|")]
        for r in body
    ]
    def _row_width(cells: list[str]) -> int:
        # Approximate the *rendered* width: strip LaTeX command tokens inside
        # math segments so e.g. "$\lvert\Delta_\text{off-target}\rvert$" is
        # not counted as 41 visible chars when the rendered glyphs are ~12.
        total = 0
        for c in cells:
            v = re.sub(r"\\[a-zA-Z]+\{?", "", c)
            v = re.sub(r"[\$\{\}\\]", "", v)
            total += len(v)
        return total + 3 * max(len(cells) - 1, 0)
    max_w = max([_row_width(cols)] + [_row_width(r) for r in body_rows])
    # The cutoff is a legibility floor, not an aesthetic one: squeezing a
    # max_w~106 tabular into one column renders near 3pt. But `table*` floats can
    # only be placed at the top of a page or on a dedicated float page, so an
    # over-eager cutoff queues them up and strands the last ones on a float page
    # of their own. Measured over the whole appendix, the widest cutoff that
    # keeps every table at >= 7pt is 58; 55 sits in the middle of the 52--58 gap
    # in the observed width distribution, so small edits cannot flip a table.
    if _PENDING_WIDE[0] or max_w > 55:
        env, resize_target = "table*", "\\textwidth"
    elif max_w > 40:
        env, resize_target = "table", "\\columnwidth"
    else:
        env, resize_target = "table", None
    _PENDING_WIDE[0] = False
    out.append(f"\\begin{{{env}}}[t]")
    # \fitwidth picks the size itself (\small -> \footnotesize -> \scriptsize), so
    # do not pre-set \small here or the step-down starts one size too low.
    out.append("\\centering" if resize_target else "\\centering\\small")
    if resize_target:
        out.append(f"\\fitwidth{{{resize_target}}}{{%")
    out.append("\\begin{tabular}{" + "".join(aligns) + "}")
    out.append("\\toprule")
    out.append(" & ".join(convert_inline(c) for c in cols) + r" \\")
    out.append("\\midrule")
    for row in body:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        out.append(" & ".join(convert_inline(c) for c in cells) + r" \\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    if resize_target:
        out.append("}")
    if _PENDING_CAPTION[0]:
        out.append(f"\\caption{{{_PENDING_CAPTION[0]}}}")
        _PENDING_CAPTION[0] = ""
    else:
        out.append("\\caption{Numerical results referenced in the surrounding text.}")
    if _PENDING_LABEL[0]:
        out.append(f"\\label{{{_PENDING_LABEL[0]}}}")
        _PENDING_LABEL[0] = ""
    else:
        out.append(f"\\label{{tab:auto{n}}}")
    out.append(f"\\end{{{env}}}")
    return out


def emit_list(env: str, items: list[str]) -> list[str]:
    """itemize/enumerate, set ragged right when the items carry file paths.

    A path has no hyphenation points, and acl.sty is globally \\sloppy, so a
    justified narrow list column pays for an unbreakable token by stretching the
    word spaces around it -- measured at up to 6.6x the body-text median in the
    released-artifacts list. Ragged right sends that slack to the margin instead.
    """
    ragged = any(
        len(code) >= 18
        for item in items
        for code in re.findall(r"`([^`]+)`", item)
    )
    out = [r"\begingroup\raggedright"] if ragged else []
    out.append(r"\begin{%s}" % env)
    out.extend(r"\item " + convert_inline(it) for it in items)
    out.append(r"\end{%s}" % env)
    if ragged:
        out.append(r"\endgroup")
    return out


def convert():
    src = SRC.read_text(encoding="utf-8").splitlines()
    scan_targets(src)
    out: list[str] = []
    i = 0
    in_code_block = False
    code_buf: list[str] = []
    code_lang = ""

    LATEX_PASSTHROUGH_ENVS = {
        "multline", "multline*",
        "equation", "equation*",
        "align", "align*",
        "gather", "gather*",
        "displaymath",
    }

    while i < len(src):
        line = src[i]

        m_env = re.match(r"^\\begin\{([^}]+)\}\s*$", line)
        if m_env and m_env.group(1) in LATEX_PASSTHROUGH_ENVS:
            env_name = m_env.group(1)
            end_pat = re.compile(rf"^\\end\{{{re.escape(env_name)}\}}\s*$")
            out.append(line)
            i += 1
            while i < len(src):
                out.append(src[i])
                if end_pat.match(src[i]):
                    i += 1
                    break
                i += 1
            continue

        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            if in_code_block:
                out.append(r"\begin{verbatim}")
                # Sanitise any non-ASCII inside verbatim — pdflatex's typewriter font
                # has no glyphs for §, →, etc. Use ASCII fallbacks.
                VERB_FIXES = {
                    "§": "Sec.",
                    "—": "--",
                    "–": "-",
                    "→": "->",
                    "≥": ">=",
                    "≤": "<=",
                    "≈": "~=",
                    "±": "+/-",
                    "…": "...",
                    "≡": "==",
                    "τ": "tau", "π": "pi", "σ": "sigma", "ρ": "rho",
                    "Δ": "Delta", "θ": "theta", "φ": "phi",
                    "↔": "<->", "✓": "[v]", "✗": "[x]",
                }
                clean = []
                for cl in code_buf:
                    for k, v in VERB_FIXES.items():
                        cl = cl.replace(k, v)
                    clean.append(cl)
                out.extend(clean)
                out.append(r"\end{verbatim}")
                code_buf = []
                in_code_block = False
                code_lang = ""
            else:
                in_code_block = True
                code_lang = m.group(1)
            i += 1
            continue

        if in_code_block:
            code_buf.append(line)
            i += 1
            continue

        # Headings — DRAFT.md uses # for paper title, ## for top-level sections, ### for subsections, #### for subsubsections
        m = re.match(r"^(#{1,4})\s+(.*?)\s*$", line)
        if m:
            level = len(m.group(1))
            # Capture the author-written number before it is stripped, so the
            # heading can carry the label that pointers at it resolve through.
            num_m = _HEADING_NUM.match(line)
            heading_num = (num_m.group(1) or num_m.group(2) or num_m.group(3)) if num_m else None
            title = convert_inline(m.group(2), refs=False)
            # Strip Section number prefixes like "1 Introduction" -> "Introduction"
            title = re.sub(r"^\d+(\.\d+)*\s+", "", title)
            # Strip "Appendix A — " prefix on appendix sections (we'll handle appendix via \appendix)
            title = re.sub(r"^Appendix\s+[A-Z]\s*[—-]+\s*", "", title)
            # Strip A.1 / A.2 / A.4.1 etc. prefixes on appendix sub(sub)section titles
            title = re.sub(r"^[A-Z]\.\d+(?:\.\d+)?\s+", "", title)
            # Strip leading "§ N.N " or "N.N.N " that some headings still carry
            title = re.sub(r"^\\S~\d+(?:\.\d+)+\s+", "", title)
            cmds = {1: None, 2: "section", 3: "subsection", 4: "subsubsection"}
            cmd = cmds.get(level)
            if cmd is None or not title:
                i += 1
                continue
            # Special-cased blocks
            tlow = title.lower()
            if tlow == "abstract":
                out.append(r"\begin{abstract}")
                out.append("@@ABSTRACT_START@@")
                i += 1
                continue
            if tlow == "references":
                # Skip the markdown References section; LaTeX uses \bibliography
                i += 1
                while i < len(src) and not re.match(r"^#{1,4}\s+", src[i]):
                    i += 1
                continue
            out.append(f"\\{cmd}{{{title}}}")
            if heading_num:
                out.append(f"\\label{{{_label_key(heading_num)}}}")
            i += 1
            continue

        # Caption marker: <!-- caption: ... [label: tab:foo] [wide: true] -->
        # Sits on its own line just before a table or figure. Consumed silently.
        # Strip optional trailing `wide: true` flag first so it doesn't leak
        # into the caption text.
        _stripped = line
        _wide_flag = False
        m_wide = re.match(r"^(.*?)\s+wide:\s*true\s*(-->\s*)$", _stripped)
        if m_wide:
            _stripped = m_wide.group(1) + " " + m_wide.group(2)
            _wide_flag = True
        m_cap = re.match(
            r"^\s*<!--\s*caption:\s*(.+?)\s+label:\s*(\S+)\s*-->\s*$", _stripped)
        if not m_cap:
            m_cap = re.match(r"^\s*<!--\s*caption:\s*(.+?)\s*-->\s*$", _stripped)
        if m_cap:
            _PENDING_CAPTION[0] = convert_inline(m_cap.group(1))
            _PENDING_LABEL[0] = m_cap.group(2) if m_cap.lastindex and m_cap.lastindex >= 2 else ""
            _PENDING_WIDE[0] = _wide_flag
            i += 1
            continue

        # External figure include: <!-- figure: <stem> --> on its own line.
        # Emits \input{figures/<stem>.tex} so the figure file is rendered in place.
        m_fig = re.match(r"^\s*<!--\s*figure:\s*(\S+)\s*-->\s*$", line)
        if m_fig:
            out.append(f"\\input{{figures/{m_fig.group(1)}.tex}}")
            i += 1
            continue

        # Tables (line starts with | and next line is separator with dashes).
        # Tolerate a stray <!-- caption: ... --> marker placed BETWEEN the
        # separator and the first body row (common in our markdown drafts);
        # consume it and continue accumulating table rows.
        if line.lstrip().startswith("|") and i + 1 < len(src) and re.match(r"^\s*\|[\s:|-]+\|\s*$", src[i + 1]):
            tbl = [src[i], src[i + 1]]
            i += 2
            while i < len(src):
                cur = src[i]
                if cur.lstrip().startswith("|"):
                    tbl.append(cur)
                    i += 1
                    continue
                # Caption marker mid-table: pull it into _PENDING and skip.
                _stripped_in = cur
                _wide_flag_in = False
                m_wide_in = re.match(
                    r"^(.*?)\s+wide:\s*true\s*(-->\s*)$", _stripped_in)
                if m_wide_in:
                    _stripped_in = m_wide_in.group(1) + " " + m_wide_in.group(2)
                    _wide_flag_in = True
                m_cap_in_tbl = re.match(
                    r"^\s*<!--\s*caption:\s*(.+?)\s+label:\s*(\S+)\s*-->\s*$", _stripped_in)
                if not m_cap_in_tbl:
                    m_cap_in_tbl = re.match(
                        r"^\s*<!--\s*caption:\s*(.+?)\s*-->\s*$", _stripped_in)
                if m_cap_in_tbl:
                    _PENDING_CAPTION[0] = convert_inline(m_cap_in_tbl.group(1))
                    _PENDING_LABEL[0] = (
                        m_cap_in_tbl.group(2)
                        if m_cap_in_tbl.lastindex and m_cap_in_tbl.lastindex >= 2
                        else "")
                    _PENDING_WIDE[0] = _wide_flag_in
                    i += 1
                    continue
                break
            out.extend(convert_table(tbl))
            continue

        # Bullet lists
        if re.match(r"^\s*-\s+", line):
            items = []
            while i < len(src) and re.match(r"^\s*-\s+", src[i]):
                items.append(re.sub(r"^\s*-\s+", "", src[i]))
                i += 1
            out.extend(emit_list("itemize", items))
            continue

        # Numbered lists
        if re.match(r"^\d+\.\s+", line):
            items = []
            while i < len(src) and re.match(r"^\d+\.\s+", src[i]):
                items.append(re.sub(r"^\d+\.\s+", "", src[i]))
                i += 1
            out.extend(emit_list("enumerate", items))
            continue

        # Horizontal rule → just blank
        if re.match(r"^---+\s*$", line):
            out.append("")
            i += 1
            continue

        # Block quote
        if line.lstrip().startswith(">"):
            out.append(r"\begin{quote}")
            while i < len(src) and src[i].lstrip().startswith(">"):
                content = re.sub(r"^\s*>\s*", "", src[i])
                out.append(convert_inline(content))
                i += 1
            out.append(r"\end{quote}")
            continue

        # Plain line
        if line.strip():
            # A numbered run-in label is the anchor its pointers link to.
            # \phantomsection is what makes the target land on this paragraph
            # rather than on the enclosing heading.
            rl = _RUNIN_NUM.match(line)
            self_num = None
            if rl and rl.group(1) in _PSEUDO_TARGETS:
                out.append(f"\\phantomsection\\label{{{_PSEUDO_TARGETS[rl.group(1)]}}}")
                self_num = rl.group(1)
            out.append(convert_inline(line, self_num=self_num))
        else:
            out.append("")
        i += 1

    body = "\n".join(out) + "\n"
    # Insert \end{abstract} before the first \section after @@ABSTRACT_START@@
    if "@@ABSTRACT_START@@" in body:
        idx = body.index("@@ABSTRACT_START@@")
        rest = body[idx + len("@@ABSTRACT_START@@"):]
        # find first \section{
        m = re.search(r"\\section\{", rest)
        if m:
            cut = m.start()
            body = (
                body[:idx]
                + rest[:cut]
                + r"\end{abstract}" + "\n\n"
                + rest[cut:]
            )
        else:
            body = body.replace("@@ABSTRACT_START@@", "")
    # Split body: main + appendix at the first \section{Implementation details}
    marker = "\\section{" + APPENDIX_TITLE_KEYWORD + "}"
    if marker in body:
        cut = body.index(marker)
        main_part, appendix_part = body[:cut].rstrip() + "\n", body[cut:]
    else:
        main_part, appendix_part = body, ""

    # Extract POST_REFS_SECTIONS (Limitations, Ethics) from main_part. Each
    # section runs from its \section{...} line up to (but not including) the
    # next \section{...}. We move them to post_refs.tex and demote their
    # heading to \section*{...} so they appear unnumbered after the bibliography.
    extracted = []
    for title in POST_REFS_SECTIONS:
        start_marker = f"\\section{{{title}}}"
        if start_marker not in main_part:
            continue
        start = main_part.index(start_marker)
        # find next \section{...}
        rest = main_part[start + len(start_marker):]
        m = re.search(r"\\section\{", rest)
        if m:
            end = start + len(start_marker) + m.start()
        else:
            end = len(main_part)
        block = main_part[start:end].rstrip() + "\n\n"
        # demote \section -> \section* so it's unnumbered post-references
        block = block.replace(start_marker, f"\\section*{{{title}}}", 1)
        # The number this section had in the draft is gone once it is starred, so
        # drop the numeric label and anchor it by name instead. \phantomsection is
        # required for the link to land here and not on the last numbered section.
        block = re.sub(r"\\label\{sec:\d+\}\n", "", block, count=1)
        block = block.replace(
            f"\\section*{{{title}}}",
            f"\\phantomsection\\label{{{UNNUMBERED_LABELS[title]}}}\n\\section*{{{title}}}", 1)
        extracted.append(block)
        # splice out of main_part (leave a blank line so paragraph flow stays)
        main_part = (main_part[:start] + main_part[end:]).rstrip() + "\n\n"
    post_refs_part = "".join(extracted)

    # Post-process: render P^2 with negative kern so pdftotext outputs "P2"
    # (matches submitted PDF convention; default $P^2$ extracts as "P 2").
    # Also map raw Unicode △ to $\triangle$ so it renders (default tectonic
    # config has no glyph for U+25B3 in text).
    def post_fix(s: str) -> str:
        s = s.replace("$P^2$", "$P^{\\!2}$")
        # The Unicode path (P² -> P$^{2}$ via the character map) has to land on the
        # same negative-kern form, otherwise the two spellings render differently
        # and pdftotext splits one of them as "P 2". The submitted PDF applied this
        # by hand and missed 11 appendix occurrences; normalising here fixes the
        # inconsistency SUBMISSION_STATE.md records.
        s = s.replace("P$^{2}$", "$P^{\\!2}$")
        s = s.replace("△", "$\\triangle$")
        # A score out of ten reads as a fraction when the slash carries spaces.
        # Only tighten it when the "10" closes the math span: "4 / 10 conditions"
        # and "0 / 5 / 10" are prose and stay as they are.
        s = s.replace(" / 10$", "/10$")
        # Brace-grouping stops the relation being set with binary-operator spacing,
        # which is too loose inside running text. Applied only where the submitted
        # PDF did: $K=15$ occurs three times and was tightened in just one of them,
        # so the plain form is the majority spelling and stays.
        s = s.replace("$P=R=1.00$", "$P{=}R{=}1.00$")
        # Stop a line break between the digit and the en dash of a numeric range.
        s = s.replace("26--39-atom", "26\\nobreakdash--39-atom")
        # A.4.1 is referenced as \S in the draft's five other mentions, and the
        # submitted PDF matched this sixth one to them. Written against the
        # converted form because convert_refs has already run by this point.
        s = s.replace("App.~\\ref{app:A.4.1}", "\\S~\\ref{app:A.4.1}")
        return s
    main_part = post_fix(main_part)
    post_refs_part = post_fix(post_refs_part)
    appendix_part = post_fix(appendix_part)

    MAIN_DST.write_text(main_part, encoding="utf-8")
    POST_REFS_DST.write_text(post_refs_part, encoding="utf-8")
    APPENDIX_DST.write_text(appendix_part, encoding="utf-8")
    print(f"Wrote {MAIN_DST} ({main_part.count(chr(10))} lines)")
    print(f"Wrote {POST_REFS_DST} ({post_refs_part.count(chr(10))} lines)")
    print(f"Wrote {APPENDIX_DST} ({appendix_part.count(chr(10))} lines)")
    print(f"Refs: {_REF_STATS['ref']} \\ref + {_REF_STATS['hyperref']} \\hyperref "
          f"({len(_REF_TARGETS)} numbered targets, {len(_PSEUDO_TARGETS)} run-in labels)")
    if _REF_UNRESOLVED:
        total = sum(_REF_UNRESOLVED.values())
        print(f"WARNING: {total} pointer(s) left as literal text — they point at no "
              f"section and will not follow renumbering:")
        for target, n in _REF_UNRESOLVED.most_common():
            print(f"  {target}  ×{n}")
    misses, in_titles = audit_literal_pointers(main_part, post_refs_part, appendix_part)
    if misses:
        print(f"WARNING: {sum(misses.values())} pointer(s) still literal in the output — "
              f"the pattern never matched them, so they are not links:")
        for num, n in misses.most_common():
            print(f"  {num}  ×{n}")
    else:
        print(f"Literal pointers unaccounted for: 0 "
              f"({sum(_REF_ANCHORED.values())} run-in label(s) naming themselves, "
              f"{sum(in_titles.values())} in heading(s) — both intentional)")


if __name__ == "__main__":
    convert()
