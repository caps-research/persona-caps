# Experiment Plan

## Phase 0 — Scaffold (this commit)

- [x] Project structure
- [x] Multi-backend LLM client (OpenAI, Anthropic, MiniMax-M2.7)
- [x] If-then corpus loader
- [x] Embedding-based retriever
- [x] Minimal CAPS-RAG agent
- [x] Inventory administration script
- [x] Seed-corpus generation script
- [x] One example corpus pair (high/low neuroticism)
- [x] One example inventory (IPIP short)

## Phase 1 — RQ1: Stable measurable personality

**Goal:** Show that an if-then corpus produces a stable, measurable personality profile under a standard inventory.

- Generate corpora for the 5 Big Five dimensions × {high, low} = 10 corpora, ~30 atoms each
- Administer IPIP-NEO short (60 items) to each agent under each backend
- Measure: factor scores, internal consistency (Cronbach α), test-retest reliability across 5 runs
- Baseline: trait-prompt ("You are highly conscientious")
- **Pass criterion**: target dimension significantly different (p<.01) and internal α > .70

## Phase 2 — RQ2: Cross-situational consistency (the CAPS test)

**Goal:** Show situation-behavior signatures, not just trait-level consistency.

- Construct ~20 base situations × 3 paraphrases each
- Same agent responds to all paraphrases
- Measure: behavioral similarity within paraphrase-set (high), across base-situations (lower)
- This is the empirical signature CAPS predicts and trait theory cannot explain

## Phase 3 — RQ3: vs baselines

Compare:
1. Zero-shot trait-prompt
2. Long persona description (SOUL.md style)
3. Random demonstration baseline (few-shot control)
4. **persona-caps (ours)**

Metrics: inventory consistency, cross-situational signature, qualitative behavioral coherence (human eval, n≥3 raters).

## Phase 4 — RQ4: Targeted editing

- Start with neutral corpus
- Add 5 high-conscientiousness atoms → re-administer inventory
- Measure: target dimension shift, side-effects on other dimensions
- IRiS cannot do this trivially; this is our killer demo

## Phase 5 — RQ5: Cross-model transfer

Same corpus, three backends (GPT-4o, Claude, MiniMax-M2.7). Pearson r of factor scores across models. >.7 = transferable.

## Phase 6 — User study (optional, for CHI submission)

Users author their own if-then atoms via a card interface. Compare against writing a persona description. Measure: ease, perceived control, behavioral fit.

---

## Stretch / future extensions

- Replace `(situation, behavior)` atoms with full `(situation, encoding, expectancy, affect, goal, plan, behavior)` Cognitive-Affective Units (CAUs) per Mischel
- Hook into mahm (memory-augmented agent line of work in this workspace) — let the corpus *grow* through experience. This becomes "personality development via memory."
