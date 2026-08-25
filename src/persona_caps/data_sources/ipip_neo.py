"""IPIP-NEO-300 loader (agent-track corpus source).

Per CONSENSUS §2 dual-track rule, this source is **forbidden in paper
main results** and may only feed atoms into ``data/if_then/agent/...``.
It exists so we can build a downstream agent-experiments toolkit on the
public-domain IPIP-NEO-300 item bank.

This module:

  1. Downloads (and caches under ``data/raw/ipip_neo_300/``)
     - ``questions.json`` -- the 300 items from the public mirror
       maintained by NeuroQuestAi
     - ``facets_key.html`` -- the IPIP-ORI 'NEO Facets Key' page
       (https://ipip.ori.org/newNEOFacetsKey.htm)
  2. Parses both into a unified per-item record:
       ``{id, text, domain, facet, polarity}``
     where domain is one of the Big Five and polarity is ``"+"`` or
     ``"-"`` (whether the item, scored as written, indexes the high or
     low pole of the facet).

The facet-key HTML is small and mostly stable. We parse it heuristically
(table-of-items between facet headers like ``N1: ANXIETY``).
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

QUESTIONS_URL = (
    "https://raw.githubusercontent.com/NeuroQuestAi/five-factor-e/"
    "main/data/IPIP-NEO/300/questions.json"
)
FACETS_KEY_URL = "https://ipip.ori.org/newNEOFacetsKey.htm"

CACHE_DIR = Path("data/raw/ipip_neo_300")

# Facet code -> domain (Big Five).
FACET_TO_DOMAIN: dict[str, str] = {
    "N1": "neuroticism",
    "N2": "neuroticism",
    "N3": "neuroticism",
    "N4": "neuroticism",
    "N5": "neuroticism",
    "N6": "neuroticism",
    "E1": "extraversion",
    "E2": "extraversion",
    "E3": "extraversion",
    "E4": "extraversion",
    "E5": "extraversion",
    "E6": "extraversion",
    "O1": "openness",
    "O2": "openness",
    "O3": "openness",
    "O4": "openness",
    "O5": "openness",
    "O6": "openness",
    "A1": "agreeableness",
    "A2": "agreeableness",
    "A3": "agreeableness",
    "A4": "agreeableness",
    "A5": "agreeableness",
    "A6": "agreeableness",
    "C1": "conscientiousness",
    "C2": "conscientiousness",
    "C3": "conscientiousness",
    "C4": "conscientiousness",
    "C5": "conscientiousness",
    "C6": "conscientiousness",
}

FACET_NAMES: dict[str, str] = {
    "N1": "anxiety",
    "N2": "anger",
    "N3": "depression",
    "N4": "self-consciousness",
    "N5": "immoderation",
    "N6": "vulnerability",
    "E1": "friendliness",
    "E2": "gregariousness",
    "E3": "assertiveness",
    "E4": "activity level",
    "E5": "excitement-seeking",
    "E6": "cheerfulness",
    "O1": "imagination",
    "O2": "artistic interests",
    "O3": "emotionality",
    "O4": "adventurousness",
    "O5": "intellect",
    "O6": "liberalism",
    "A1": "trust",
    "A2": "morality",
    "A3": "altruism",
    "A4": "cooperation",
    "A5": "modesty",
    "A6": "sympathy",
    "C1": "self-efficacy",
    "C2": "orderliness",
    "C3": "dutifulness",
    "C4": "achievement-striving",
    "C5": "self-discipline",
    "C6": "cautiousness",
}


@dataclass(frozen=True)
class IpipItem:
    item_id: int
    text: str
    domain: str
    facet_code: str
    facet_name: str
    polarity: str  # "+" or "-"

    def to_provenance(self) -> dict:
        return {
            "source": "IPIP-NEO-300",
            "item_id": self.item_id,
            "domain": self.domain,
            "facet_code": self.facet_code,
            "facet_name": self.facet_name,
            "polarity_in_inventory": self.polarity,
            "license": "public_domain (https://ipip.ori.org/newPermission.htm)",
            "track": "agent",
        }


# ---------------------------------------------------------------------------
# Download / cache helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "persona-caps/0.1"})
    with urllib.request.urlopen(req) as r:
        dest.write_bytes(r.read())
    return dest


def load_questions(cache_dir: Path = CACHE_DIR) -> dict[int, str]:
    path = _download(QUESTIONS_URL, cache_dir / "questions.json")
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for q in obj["questions"]:
        out[int(q["id"])] = q["text"].strip()
    return out


# ---------------------------------------------------------------------------
# Facet-key HTML parser
# ---------------------------------------------------------------------------

# Lines that look like '| + keyed | Worry about things. |' or just
# '| Worry about things. |' or '|  keyed | Am not easily bothered by things. |'.

_LINE_KEY_HEADER_RE = re.compile(r"\|\s*([+\-]?)\s*keyed\s*\|\s*(.+?)\s*\|")
_LINE_PLAIN_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*$")
_FACET_HEADER_RE = re.compile(r"\|\s*([NEOAC]\d):\s*[A-Z\- ]+\(.*?\)\s*\|")


def parse_facets_key(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse the IPIP-ORI 'NEO Facets Key' markdown-table-ish text.

    Returns ``{facet_code: [(item_text, polarity), ...]}`` for all 30
    facets, 10 items each.
    """
    facets: dict[str, list[tuple[str, str]]] = {}
    current_items: list[tuple[str, str]] = []
    current_polarity: str | None = None  # "+" or "-"
    lines = [ln.strip() for ln in text.splitlines()]

    for line in lines:
        if not line:
            continue

        m_facet = _FACET_HEADER_RE.search(line)
        if m_facet:
            code = m_facet.group(1)
            facets[code] = current_items
            current_items = []
            current_polarity = None
            continue

        m_kh = _LINE_KEY_HEADER_RE.search(line)
        if m_kh:
            mark = m_kh.group(1)
            current_polarity = "+" if mark == "+" else "-"
            item = m_kh.group(2).strip()
            if item.lower() != "keyed":
                current_items.append((_clean(item), current_polarity))
            continue

        m_p = _LINE_PLAIN_RE.match(line)
        if m_p:
            content = m_p.group(1).strip()
            if not content or content == "---":
                continue
            if current_polarity is None:
                continue
            current_items.append((_clean(content), current_polarity))
            continue
    return facets


def _clean(s: str) -> str:
    """Normalise an IPIP item text for matching against questions.json."""
    # Some items in the facets page use curly quotes; questions.json uses straight ones.
    return (
        s.replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .strip()
        .rstrip(".")
        .strip()
    )


def _normalise_for_matching(s: str) -> str:
    """Aggressive normalisation: lowercase + drop leading copula-like words.

    The IPIP-ORI facets page renders some items as 'Am willing to try
    anything once' while questions.json drops the leading copula
    ('Willing to try anything once'). We match either form.
    """
    out = _clean(s).lower()
    for prefix in ("am ", "have ", "do ", "i "):
        if out.startswith(prefix):
            out = out[len(prefix):]
            break
    return out


def load_facets_key(cache_dir: Path = CACHE_DIR) -> dict[str, list[tuple[str, str]]]:
    """Download (cached) the IPIP-ORI facets key and parse it.

    The HTML version on ipip.ori.org is messy. We allow the user to
    drop a hand-cleaned ``facets_key.txt`` (markdown-table form) into
    the cache dir; if present, we use that instead. (Our
    ``scripts/seed_ipip_facets_key.py`` writes that file.)
    """
    txt_path = cache_dir / "facets_key.txt"
    if txt_path.exists():
        return parse_facets_key(txt_path.read_text(encoding="utf-8"))
    html_path = _download(FACETS_KEY_URL, cache_dir / "facets_key.html")
    return parse_facets_key(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Public loader: assemble per-item records
# ---------------------------------------------------------------------------


def load_items(cache_dir: Path = CACHE_DIR) -> list[IpipItem]:
    questions = load_questions(cache_dir)
    facets = load_facets_key(cache_dir)

    text_to_qid: dict[str, int] = {_clean(t): qid for qid, t in questions.items()}
    norm_to_qid: dict[str, int] = {
        _normalise_for_matching(t): qid for qid, t in questions.items()
    }
    items: list[IpipItem] = []
    matched_qids: set[int] = set()
    unmatched: list[tuple[str, str, str]] = []  # (facet, item_text, polarity)

    for facet_code, lst in facets.items():
        if facet_code not in FACET_TO_DOMAIN:
            continue
        for item_text, polarity in lst:
            qid = text_to_qid.get(_clean(item_text))
            if qid is None:
                qid = norm_to_qid.get(_normalise_for_matching(item_text))
            if qid is None:
                qid = _fuzzy_match(item_text, text_to_qid)
            if qid is None:
                unmatched.append((facet_code, item_text, polarity))
                continue
            if qid in matched_qids:
                # An item assigned to two facets; allowed but warn.
                pass
            matched_qids.add(qid)
            items.append(
                IpipItem(
                    item_id=qid,
                    text=questions[qid],
                    domain=FACET_TO_DOMAIN[facet_code],
                    facet_code=facet_code,
                    facet_name=FACET_NAMES[facet_code],
                    polarity=polarity,
                )
            )
    if unmatched:
        msg = (
            f"Failed to match {len(unmatched)} IPIP item(s) between facets-key "
            f"and questions.json. First few: {unmatched[:5]}. "
            "If this happens, drop a hand-cleaned facets_key.txt into "
            f"{cache_dir} (see seed_ipip_facets_key.py)."
        )
        raise RuntimeError(msg)
    return items


def _fuzzy_match(needle: str, haystack: dict[str, int]) -> int | None:
    """Try a token-set comparison if exact match failed."""
    needle_tokens = set(re.findall(r"\w+", needle.lower()))
    if not needle_tokens:
        return None
    best_qid = None
    best_score = 0.0
    for k, qid in haystack.items():
        ks = set(re.findall(r"\w+", k.lower()))
        if not ks:
            continue
        score = len(needle_tokens & ks) / max(len(needle_tokens), len(ks))
        if score > best_score:
            best_score = score
            best_qid = qid
    if best_score >= 0.85:
        return best_qid
    return None
