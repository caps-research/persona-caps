"""Smoke test for the conditional-retrieval machinery.

No LLM calls. Purely verifies:
  (1) Retriever(mode="argmax") is deterministic (same seed -> same atoms).
  (2) Retriever(mode="softmax", tau small) is *almost* always argmax-equivalent.
  (3) Retriever(mode="softmax", tau large) actually samples differently
      across runs (i.e. is genuinely random).
  (4) JointCapsAgent retrieves from every active cell (not just one).
  (5) JointCapsAgent with weight=0 on a cell drops that cell from the mixture.
  (6) JointCapsAgent.sample_cells=True respects dim_weights as the marginal.

Exit code 0 on full pass, 1 on any failure.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from persona_caps.agent import JointCapsAgent  # noqa: E402
from persona_caps.corpus import load_corpus_paper_only  # noqa: E402
from persona_caps.retriever import Retriever  # noqa: E402

CORPUS_DIR = ROOT / "data/if_then/paper/beck_2022_v2"
QUERY = (
    "Below is a personality questionnaire item. Item: "
    '"I have frequent mood swings."'
)


class FakeLLM:
    """Stand-in LLM that records the system prompt instead of calling out."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, messages, system=None, **kw):
        self.calls.append({"messages": messages, "system": system})
        return "RATING: 3\nWHY: stub response"


def _check(label: str, ok: bool, detail: str = "") -> None:
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        global _failed
        _failed = True


_failed = False


def main() -> int:
    print("=" * 72)
    print("Conditional-retrieval smoke test")
    print("=" * 72)

    print("\nLoading corpora...")
    corpus_hN = load_corpus_paper_only(CORPUS_DIR / "high_neuroticism.jsonl")
    corpus_lE = load_corpus_paper_only(CORPUS_DIR / "low_extraversion.jsonl")
    corpus_hC = load_corpus_paper_only(CORPUS_DIR / "high_conscientiousness.jsonl")
    corpus_hA = load_corpus_paper_only(CORPUS_DIR / "high_agreeableness.jsonl")
    corpus_hO = load_corpus_paper_only(CORPUS_DIR / "high_openness.jsonl")
    print(f"  high_N: {len(corpus_hN)} atoms")
    print(f"  low_E:  {len(corpus_lE)} atoms")
    print(f"  high_C: {len(corpus_hC)} atoms")
    print(f"  high_A: {len(corpus_hA)} atoms")
    print(f"  high_O: {len(corpus_hO)} atoms")

    # ----- (1) argmax determinism -----
    print("\n(1) argmax mode is deterministic across calls and seeds")
    r1 = Retriever(corpus_hN, mode="argmax", rng=np.random.default_rng(1))
    r2 = Retriever(corpus_hN, mode="argmax", rng=np.random.default_rng(99))
    out1 = [a.id for a, _ in r1.retrieve(QUERY, k=4)]
    out2 = [a.id for a, _ in r2.retrieve(QUERY, k=4)]
    out3 = [a.id for a, _ in r1.retrieve(QUERY, k=4)]
    _check("two seeds, same result", out1 == out2, f"out1={out1}\n      out2={out2}")
    _check("same retriever twice, same result", out1 == out3)

    # ----- (2) softmax tau very small ~ argmax -----
    print("\n(2) softmax with tiny tau closely tracks argmax (top-1 must agree)")
    rs = Retriever(corpus_hN, mode="softmax", tau=0.001, rng=np.random.default_rng(7))
    out_sm = [a.id for a, _ in rs.retrieve(QUERY, k=4)]
    _check(
        "softmax(tau=0.001) top-1 == argmax top-1",
        out_sm[0] == out1[0],
        f"sm_top1={out_sm[0]} vs argmax_top1={out1[0]}",
    )

    # ----- (3) softmax tau large is genuinely random -----
    print("\n(3) softmax with tau=2.0 actually varies the chosen atoms across seeds")
    seen_top1: set = set()
    for s in range(20):
        rs = Retriever(corpus_hN, mode="softmax", tau=2.0, rng=np.random.default_rng(s))
        out = [a.id for a, _ in rs.retrieve(QUERY, k=4)]
        seen_top1.add(out[0])
    _check(
        "softmax(tau=2.0) sees more than one distinct top-1 across 20 seeds",
        len(seen_top1) >= 3,
        f"distinct top-1 atoms: {len(seen_top1)} -> {sorted(seen_top1)[:5]}...",
    )

    # ----- (4) JointCapsAgent retrieves from every active cell -----
    print("\n(4) JointCapsAgent fires atoms from every active cell")
    cells = {
        "high_neuroticism":      corpus_hN,
        "low_extraversion":      corpus_lE,
        "high_conscientiousness": corpus_hC,
        "high_agreeableness":    corpus_hA,
        "high_openness":         corpus_hO,
    }
    fake_llm = FakeLLM()
    agent = JointCapsAgent(
        cells=cells,
        llm=fake_llm,
        top_k_per_dim=2,
        mode="argmax",
        rng=np.random.default_rng(0),
    )
    resp = agent.respond(QUERY)
    fired_ids = {a.id for a, _ in resp.fired_atoms}
    n_fired = len(resp.fired_atoms)
    _check(
        "fired exactly 5 cells x 2 atoms = 10 atoms (no dup expected across cells)",
        n_fired == 10,
        f"got n_fired={n_fired}",
    )
    # check coverage: at least one atom from each of the 5 cells
    cell_origins: Counter = Counter()
    for atom, _ in resp.fired_atoms:
        for cname, c in cells.items():
            if atom in c.atoms:
                cell_origins[cname] += 1
                break
    _check(
        "all 5 cells contributed at least one atom",
        len(cell_origins) == 5,
        f"cell coverage = {dict(cell_origins)}",
    )

    # ----- (5) zero-weight cell is dropped -----
    print("\n(5) JointCapsAgent with one cell weighted 0 drops that cell")
    weights = {n: 1.0 for n in cells}
    weights["high_openness"] = 0.0
    fake_llm2 = FakeLLM()
    agent2 = JointCapsAgent(
        cells=cells,
        llm=fake_llm2,
        dim_weights=weights,
        top_k_per_dim=2,
        mode="argmax",
        rng=np.random.default_rng(0),
    )
    resp2 = agent2.respond(QUERY)
    fired2 = resp2.fired_atoms
    cell_origins2: Counter = Counter()
    for atom, _ in fired2:
        for cname, c in cells.items():
            if atom in c.atoms:
                cell_origins2[cname] += 1
                break
    _check(
        "high_openness contributes 0 atoms (weight=0)",
        cell_origins2.get("high_openness", 0) == 0,
        f"cell coverage = {dict(cell_origins2)}",
    )
    _check(
        "remaining 4 cells together fire 4 x 2 = 8 atoms",
        len(fired2) == 8,
        f"got n_fired={len(fired2)}",
    )

    # ----- (6) sample_cells respects weights as marginal frequency -----
    print("\n(6) sample_cells=True respects dim_weights as the marginal frequency")
    cells_two = {"high_neuroticism": corpus_hN, "low_extraversion": corpus_lE}
    weights_skew = {"high_neuroticism": 0.9, "low_extraversion": 0.1}
    fake_llm3 = FakeLLM()
    agent3 = JointCapsAgent(
        cells=cells_two,
        llm=fake_llm3,
        dim_weights=weights_skew,
        top_k_per_dim=1,
        mode="argmax",
        sample_cells=True,
        cells_per_step=1,
        rng=np.random.default_rng(123),
    )
    counts = Counter()
    for _ in range(200):
        resp = agent3.respond(QUERY)
        for atom, _ in resp.fired_atoms:
            for cname, c in cells_two.items():
                if atom in c.atoms:
                    counts[cname] += 1
                    break
    n_total = sum(counts.values())
    high_n_share = counts["high_neuroticism"] / max(n_total, 1)
    _check(
        "approximate marginal: high_N share is in [0.78, 0.97] for w=0.9",
        0.78 <= high_n_share <= 0.97,
        f"observed share = {high_n_share:.2f} ({counts})",
    )

    print("\n" + "=" * 72)
    if _failed:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS — conditional-retrieval machinery is wired correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
