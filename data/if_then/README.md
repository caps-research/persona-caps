# `data/if_then/` — corpus directory layout

Three tiers of provenance, governed by `CONSENSUS.md` §2 (dual-track rule).
**Paper main results may only be computed from `paper/`.** All other
directories are for sandbox / smoke-testing / agent-toolkit use.

```
data/if_then/
  paper/           # paper-track only. Empirically derived from real
                   # behavioural data. Each atom MUST carry a populated
                   # `provenance` field.
    beck_2022/     # openESM 0060 (Beck et al. 2022) ESM dataset
                   # — current primary source. See:
                   #   scripts/build_paper_atoms_beck.py
                   #   src/persona_caps/data_sources/beck2022.py
                   # 177 atoms across 10 (dim, polarity) cells.

  agent/           # agent-track only. May derive from inventory item
                   # banks (circular by construction). Forbidden in
                   # paper main results — enforced by
                   # `persona_caps.load_corpus_paper_only()`.
    ipip_neo_300/  # IPIP-NEO-300 (300 items × ~1 atom each)
                   # — public domain. See:
                   #   scripts/build_agent_atoms_ipip.py
                   #   src/persona_caps/data_sources/ipip_neo.py

  *.jsonl          # legacy hand-authored seeds (10 atoms per cell, 5
                   # dims × 2 polarities + 1 noise file). Predate the
                   # dual-track rule. Treated as `seed/` tier:
                   #   - kept for backwards-compat smoke runs
                   #   - NOT a valid paper-track source
                   # Will be moved to `seed/` in a follow-up cleanup.
```

## Loading rules

```python
from persona_caps import load_corpus, load_corpus_paper_only

# anywhere — permissive
c = load_corpus("data/if_then/agent/ipip_neo_300/high_neuroticism.jsonl")

# paper main experiments — refuses agent/ paths
c = load_corpus_paper_only("data/if_then/paper/beck_2022/high_neuroticism.jsonl")
c = load_corpus_paper_only("data/if_then/agent/...")  # raises PermissionError
```

## Provenance schema

`paper/` atoms carry a `provenance` dict with:

```jsonc
{
  "dataset_id": "0060_beck",            // openESM dataset id (or other source)
  "dataset_version": "1.0.0",
  "target_dimension": "neuroticism",    // dim being mined
  "polarity": "high",                   // pole of trait
  "situation_key": "lonely",            // raw situation column
  "behavior_key": "depressed",          // raw behavior column
  "behavior_construct": "depression, ...",
  "behavior_polarity_within_dim": "positive",  // or "reverse" -- whether the raw item is reverse-keyed for its construct's dim
  "effect_size_cohens_d": 1.247,
  "n_situation_active": 47,
  "n_situation_inactive": 359,
  "mean_active": 3.809,
  "mean_inactive": 2.479,
  "n_persons_in_group": 51
}
```

`agent/` atoms carry their own provenance (inventory item id, facet code,
license, `track: "agent"`).

`seed/` (legacy) atoms have `provenance: null`.
