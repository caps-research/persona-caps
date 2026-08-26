# Persona-CAPS

Code and data for *Persona-CAPS: Controllable and Auditable LLM Personality via
Retrieval over Situation–Behavior Contingencies* (EMNLP 2026).

Guanghao Liu, Guangzhou An, Yang Xu, YU CHEN.

This archive is **end-to-end self-contained**: code, evaluation harness, frozen run logs (under `runs/`), and a small unit-test suite. All result tables and figures in the paper can be regenerated from `runs/` without re-querying any commercial API; the experiment scripts are also runnable from scratch given API keys.

---

## 1. Layout

```
src/persona_caps/         Core package
  corpus.py                  load / manage if-then JSONL corpora
  retriever.py               top-k cosine retrieval (all-MiniLM-L6-v2)
  agent.py                   render retrieved atoms into a label-free system prompt
  llm.py                     multi-backend client (OpenAI / Gemini / open-weight)
  eval.py                    Mini-IPIP / PersonalityBench scoring
  situations.py              situation pools for §4.2 cross-situational eval
  data_sources/              ESM-data and IPIP-NEO loaders

experiments/              Paper-section entry points (see §3 below)
scripts/                  Corpus mining, leakage checks, aggregation, plotting
runs/                     Frozen JSONL result logs referenced in the paper
data/                     Released CAPS corpus + Mini-IPIP / PB stimuli
                          (also distributed separately as the Data archive)
tests/                    Unit tests for corpus / retriever / agent
docs/                     Internal design notes (not required for reproduction)
requirements.txt          Python dependencies
pyproject.toml            Editable-install metadata
.env.example              Template for API keys (copy to .env)
```

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env       # fill in OPENAI_API_KEY etc., or skip and reuse runs/
```

Tested on Python 3.10–3.12. CPU is sufficient for retrieval; subject-side LLMs are queried via API (closed) or a local OpenAI-compatible server (open-weight).

## 3. Reproduction map (paper section → entry point)

| § | Topic | Entry point |
| :-- | :-- | :-- |
| 3 | ESM mining pipeline (166-atom corpus) | `scripts/build_paper_atoms_beck.py` |
| 4.1 | Mini-IPIP trait fidelity (5 conditions × 3 vendors) | `experiments/exp01_consistency.py` |
| 4.2 | Content-specificity ablations (`random_demos` / `wrong_trait`) | `experiments/exp01_consistency.py` (ablation arms) |
| 4.3 | Compositional editing case study (Fig. 2) | `experiments/exp03_editing.py` |
| 4.4 | PersonalityBench 3-backbone × 3-method head-to-head | `experiments/exp_personalitybench.py` → `experiments/judge_personalitybench.py` |
| 4.5 | 1-D mixture-weight (π) sweep | `experiments/exp_pi_sweep.py` (CAPS) / `experiments/exp_pi_sweep_promptbase.py` (P² baseline) |
| 4.6 | 5-D joint composition on 12 deployment-realistic profiles | `experiments/exp_p2_joint_5d.py`, `experiments/exp_rqg_generalization.py` |
| 4.7 | Audit-loop study — pollution sanity gate, universal-recall sweep, sparse-edit triage, P²-sanity baseline, GPT-4o cross-vendor | `scripts/audit_loop/inject_polarity_flips.py` (sanity gate) → `scripts/audit_loop/run_sweep.py` (universal-recall) / `scripts/audit_loop/locate_via_responses.py` (sparse-edit) → `scripts/audit_loop/run_fix_sweep.py` (atom-level repair) → `scripts/audit_loop/run_p2_sweep.py` (P² baseline) → `scripts/audit_loop/run_xvendor_pilot.py` (GPT-4o cross-vendor); aggregation via `scripts/audit_loop/aggregate.py` and `scripts/audit_loop/_paper_stats.py` (logs in `runs/audit_loop/`, `runs/audit_loop_xvendor/`) |
| App. A.2, B.1 | Cross-situational evaluation set and its $W - A$ metric-level null | `experiments/exp02_cross_situational.py` |
| App. A.4 | No-leakage validation (trigram-Jaccard) | `scripts/npti_overlap_diagnostic.py` |
| App. A.10.1 | Cross-family judge replication (r = +0.951) | `scripts/analyze_cross_judge_agreement.py` |
| App. C.1–C.3 | Retrieval-temperature and mixture sweeps (Sweeps A–C) | `experiments/exp_conditional_retrieval.py`, `experiments/exp_conditional_persona.py` |
| App. (open-weight) | Llama-3-8B / Qwen-3 backbone runs | `scripts/run_llama3_8b_full.sh`, `scripts/run_pi_sweep_qwen35_9b.sh` |

A single-command reproduction over the canonical 3-backbone × 3-method PersonalityBench grid:

```bash
bash scripts/run_pb_3backbone_3method.sh
python scripts/aggregate_pb_3x3.py
```

For an end-to-end batch driver covering corpus seeding + §4.1 + §4.2 across backends:

```bash
python scripts/run_all.py --backends minimax gpt-4o-mini --runs 3
```

To regenerate every aggregate referenced in Tables 1–4 directly from `runs/` (no API calls needed):

```bash
python scripts/aggregate_pb_3x3.py
python scripts/aggregate_rq4_n3.py
python scripts/aggregate_conditional_runs.py
python scripts/merge_and_bootstrap_n5.py
```

## 4. Released corpus (Data archive)

The 166-atom CAPS corpus referenced throughout the paper (§3, "166 atoms across 10 (dim × polarity) cells, mean |d| = 0.51") is shipped under `data/if_then/paper/beck_2022_v2/`, with one `.jsonl` per (Big-Five dim × polarity) cell plus a `manifest.json` of provenance hashes. All paper experiments (§4.1, §4.4, §4.5, §4.6, §4.7) use this corpus; the run logs under `runs/` reference `data/if_then/paper/beck_2022_v2` as `corpus_dir`. Mining pipeline: `scripts/build_paper_atoms_beck.py` with `min_abs_d=0.3, min_diff=0.3, top_k=20, within_dim_only=true` (manifest.json records the exact parameters). No-leakage validation against Mini-IPIP (Donnellan 2006): trigram-Jaccard ≤ 0.134 with any item, 0/20 items above 0.20 — see App. A.4 and `scripts/npti_overlap_diagnostic.py`.

The sibling directories `beck_2022/`, `beck_2022_v3/`, and `beck_2022_within/` are exploratory mining iterations preserved for reproducibility ablations only; **paper claims should be reproduced against `beck_2022_v2`**. The corresponding `runs/conditional_retrieval_v3*/` log directories contain three small Sweep B exploration files generated against `beck_2022_v3` after the paper was frozen; none of their numbers appear in the paper, and they are retained only for transparency about post-submission exploration.

## 5. Tests

```bash
pytest tests/
```

Covers atom rendering and corpus roundtrip (`tests/test_corpus.py`), ESM dialogue extraction (`tests/test_extract_dialogue.py`), and the CAPS-signature criterion of App. A.6 (`tests/test_signature.py`).

## 6. License

- **Code**: MIT (see `LICENSE`).
- **Released corpus** (`data/if_then/paper/beck_2022*`): CC BY-NC 4.0, inherited from openESM `0060_beck`.
- **Mini-IPIP item bank**: Public-domain (Donnellan et al. 2006); included verbatim under `data/`.
- **Experiment logs** (`runs/`): CC BY-NC 4.0. Many logs record prompts that quote corpus atoms
  verbatim, which makes them adapted material under the openESM terms. We apply the stricter term
  to the whole directory rather than splitting it file by file — the boundary can only be drawn by
  text matching, and an error in the permissive direction would purport to waive rights we do not
  hold. The atoms are LLM-verbalised, but the situation–behaviour pairings they express are Beck's
  empirical result, so machine rephrasing does not break the chain of derivation.

## 7. Citation

```bibtex
@inproceedings{liu-etal-2026-persona-caps,
  title     = {Persona-CAPS: Controllable and Auditable {LLM} Personality via
               Retrieval over Situation--Behavior Contingencies},
  author    = {Liu, Guanghao and An, Guangzhou and Xu, Yang and CHEN, YU},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing},
  year      = {2026}
}
```
