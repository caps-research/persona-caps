# CAPS supplemental: π-intensity sweep on PersonalityBench questions

**Date:** 2026-07-19  
**Status:** supplemental experiment (not a standard continuous-personality benchmark)

## Why this exists

There is **no** accepted test set for “continuous personality change.”  
PersonalityBench only evaluates **high vs low** poles. Graded / π-adjustable control must be tested as a **CAPS supplemental protocol**.

## Design

Reuse Extraversion *questions* as situations only (not as a continuous-trait official bench).

For each question \(q\) and \(\pi \in \{0.05,0.3,0.5,0.7,0.95\}\):

\[
k_{high}=\mathrm{round}(\pi\cdot 4),\quad
k_{low}=4-k_{high}
\]

Retrieve top-\(k_{high}\) from high-E corpus and top-\(k_{low}\) from low-E corpus by situation similarity; inject into **trait-label-free** CAPS system prompt; generate answer; MiniMax-judge Extraversion 1–5.

## Success

- Structural mix matches table for every \((q,\pi)\).  
- Mean Extraversion rating rises with \(\pi\) (endpoint Δ / Spearman).  
- CAPS prompt leak audit PASS (no trait adj / π in system prompt).  

## Claim bound

Supports **π-ordered Extraversion expression under intensity mixing** on Q&A situations.  
Does **not** create a new continuous-personality benchmark, and does not claim smooth interpolation beyond 5 discrete mix compositions at \(k=4\).
