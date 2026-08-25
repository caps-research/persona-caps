"""Beck (2022) ESM dataset loader + if-then pattern miner.

Source: openESM ``0060_beck`` (Beck et al., 2022, *Psychological Science*,
N=199 students x ~14 days x 4 beeps). The harmonized release contains:

  * ~60 state Big Five items (Likert 1-5; planned-missing per beep)
  * 8 DIAMONDS situation-perception items (1-3)
  * 18 binary context/behavior flags (in_class, argument_family, ...)
  * ~9 PANAS-style affect items every beep

Pipeline:

  1. ``load()``  -- pull dataset via ``openesm`` package (cached locally).
  2. ``trait_baselines()`` -- per-person Big Five baseline using construct
     metadata + heuristic reverse-coding.
  3. ``mine()`` -- for each (Big Five dim, polarity, situation_key,
     behavior_key) compute a Cohen's d of the within-group situation
     -> behavior coupling. Survivors with ``|d| >= 0.3`` and
     ``n_situation >= min_n`` become ``MinedPattern`` records.

The miner does NOT touch any LLM. Verbalization to natural-language
atoms is done in a separate step (``scripts/build_paper_atoms_beck.py``).
This is to keep the empirical layer cheap and re-runnable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

try:
    import openesm  # type: ignore
    import polars as pl  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "openesm + polars are required for the Beck 2022 data source. "
        "Run `pip install openesm polars`."
    ) from e


DATASET_ID = "0060_beck"
DATASET_VERSION = "1.0.0"  # the version openESM v0.1.2 currently serves


# Big Five dim canonical names used throughout the project.
BIG_FIVE = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


# Items whose phrasing is reverse-keyed against their target Big Five
# dim. Determined from the Beck 2022 codebook (the ``coding`` field of
# the openESM features table). Not all items in the dataset are listed
# here -- only the ones we actually use as Big Five state indicators.
REVERSE_KEYED: dict[str, set[str]] = {
    "neuroticism": {
        # high score on these means LOW neuroticism (emotionally stable)
        "emotionally_stable",
        "relaxed",
        "secure_comfortable",
        "controls_emotions",
        "rarely_anxious",
        "optimistic_setback",
    },
    "extraversion": {
        # high score on these means LOW extraversion
        "less_active",
        "others_charge",
        "quiet",
        "influence_hard",
        "shy",
        "rarely_excited",
    },
    "openness": {
        # high score on these means LOW openness
        "poetry_boring",
        "difficulty_imagining",
        "avoids_intellectual",
        "few_artistic",
        "little_abstract",
        "little_creativity",
    },
    "agreeableness": {
        # high score on these means LOW agreeableness
        "starts_arguments",
        "finds_fault",
        "rude",
        "little_sympathy",
        "suspicious",
        "cold",
    },
    "conscientiousness": {
        # high score on these means LOW conscientiousness
        "difficulty_starting",
        "careless",
        "messy",
        "lazy",
        "disorganized",
        "irresponsible",
    },
}


# DIAMONDS situation-perception items (continuous 1-3) and their
# canonical English phrasings (the ``coding`` field's first sentence).
DIAMONDS_SITUATIONS: dict[str, str] = {
    "mating": "potential romantic partners are present",
    "positivity": "the situation is pleasant",
    "intellect": "deep thinking is required",
    "negativity": "the situation contains negative feelings (e.g. stress, anxiety, guilt)",
    "adversity": "somebody is being threatened, accused or criticised",
    "sociality": "social interactions are possible or required",
    "duty": "there is work that has to be done",
    "deception": "somebody is being deceived",
}


# Binary context/behavior flags + canonical phrasing.
BINARY_SITUATIONS: dict[str, str] = {
    "in_class": "I am in class",
    "late": "I am late for something",
    "sick": "I feel sick",
    "procrastinating": "I am procrastinating",
    "lost_something": "I lost something",
    "bored_schoolwork": "I am bored with my schoolwork",
    "excited_schoolwork": "I am excited about my schoolwork",
    "music": "I am listening to music",
    "anxious_schoolwork": "I am anxious about schoolwork",
    "studying": "I am studying or doing schoolwork",
    "argument_family": "I am having an argument or disagreement with family",
    "argument_friend": "I am having an argument or disagreement with a friend",
    "internet": "I am on the internet",
    "tired": "I feel tired",
    "forgot_something": "I forgot something",
    "interacted_family": "I am interacting with family",
    "tv": "I am watching TV",
    "lonely": "I feel lonely",
    "interacted_friend": "I am interacting with a friend",
    "sleeping": "I am sleeping",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinedPattern:
    """One surviving (trait, polarity, situation, behavior) signature.

    ``effect_size`` is a Cohen's d of (behavior | situation_active) vs
    (behavior | situation_inactive), restricted to participants in the
    given trait-polarity group. Positive d means the behavior is more
    expressed when the situation is active.
    """

    target_dimension: str
    polarity: str  # "high" or "low"
    situation_key: str
    situation_text: str
    behavior_key: str
    behavior_description: str  # the codebook item phrasing (high pole)
    behavior_construct: str
    behavior_polarity_within_dim: str  # "positive" or "reverse"
    effect_size: float
    n_situation_active: int
    n_situation_inactive: int
    mean_active: float
    mean_inactive: float
    n_persons_in_group: int

    def to_provenance(self) -> dict:
        return {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "target_dimension": self.target_dimension,
            "polarity": self.polarity,
            "situation_key": self.situation_key,
            "behavior_key": self.behavior_key,
            "behavior_construct": self.behavior_construct,
            "behavior_polarity_within_dim": self.behavior_polarity_within_dim,
            "effect_size_cohens_d": round(self.effect_size, 3),
            "n_situation_active": self.n_situation_active,
            "n_situation_inactive": self.n_situation_inactive,
            "mean_active": round(self.mean_active, 3),
            "mean_inactive": round(self.mean_inactive, 3),
            "n_persons_in_group": self.n_persons_in_group,
        }


@dataclass
class _Item:
    name: str
    description: str
    construct: str
    target_dim: str
    is_reverse: bool


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class Beck2022:
    """Lazily loaded Beck 2022 dataset with a pattern-mining API."""

    def __init__(self) -> None:
        self._raw = None  # type: ignore[assignment]
        self._items_by_dim: dict[str, list[_Item]] | None = None

    # -- loading ---------------------------------------------------------

    def load(self):  # returns the OpenESMDataset object
        if self._raw is None:
            self._raw = openesm.get_dataset(DATASET_ID)
        return self._raw

    @property
    def data(self) -> "pl.DataFrame":
        return self.load().data

    @property
    def features(self) -> "pl.DataFrame":
        return self.load().metadata["features"]

    # -- item taxonomy ---------------------------------------------------

    def items_by_dim(self) -> dict[str, list[_Item]]:
        """Map Big Five dim -> list of state items used as indicators."""
        if self._items_by_dim is not None:
            return self._items_by_dim

        items_by_dim: dict[str, list[_Item]] = {d: [] for d in BIG_FIVE}
        feats = self.features.to_dicts()
        for feat in feats:
            name = feat.get("name") or ""
            construct = (feat.get("construct") or "").lower()
            description = feat.get("description") or ""
            if "big five" not in construct:
                continue
            target = None
            for dim in BIG_FIVE:
                if dim in construct:
                    target = dim
                    break
            # Beck 2022 has a typo "argeeableness" in one row, normalise it.
            if target is None and "argeeableness" in construct:
                target = "agreeableness"
            if target is None:
                continue
            items_by_dim[target].append(
                _Item(
                    name=name,
                    description=description,
                    construct=construct,
                    target_dim=target,
                    is_reverse=name in REVERSE_KEYED.get(target, set()),
                )
            )

        self._items_by_dim = items_by_dim
        return items_by_dim

    # -- per-person trait baseline --------------------------------------

    def trait_baselines(self) -> "pl.DataFrame":
        """Return a polars DataFrame with one row per participant and
        columns ``trait_<dim>`` in 1..5 (after reverse-coding)."""
        df = self.data
        items_by_dim = self.items_by_dim()

        agg_exprs = []
        for dim, items in items_by_dim.items():
            if not items:
                continue
            piece_exprs = []
            for item in items:
                col = pl.col(item.name)
                if item.is_reverse:
                    col = (6 - col).cast(pl.Float64)
                else:
                    col = col.cast(pl.Float64)
                piece_exprs.append(col)
            # row-wise mean across that person's responses on this dim's items
            # (planned-missing safe: sum_horizontal / count_non_null)
            sum_e = pl.sum_horizontal(piece_exprs)
            cnt_e = pl.sum_horizontal([e.is_not_null().cast(pl.Float64) for e in piece_exprs])
            row_mean = pl.when(cnt_e > 0).then(sum_e / cnt_e).otherwise(None).alias(f"_rowmean_{dim}")
            agg_exprs.append(row_mean)

        with_rowmeans = df.with_columns(agg_exprs)
        person = (
            with_rowmeans.group_by("id")
            .agg([pl.col(f"_rowmean_{dim}").mean().alias(f"trait_{dim}") for dim in BIG_FIVE])
            .sort("id")
        )
        return person

    def trait_groups(self, lo_q: float = 0.25, hi_q: float = 0.75) -> dict[str, dict[str, list[int]]]:
        """For each Big Five dim, return ``{'high': [ids], 'low': [ids]}``."""
        baselines = self.trait_baselines()
        out: dict[str, dict[str, list[int]]] = {}
        for dim in BIG_FIVE:
            col = f"trait_{dim}"
            non_null = baselines.filter(pl.col(col).is_not_null())
            if non_null.is_empty():
                out[dim] = {"high": [], "low": []}
                continue
            lo = non_null[col].quantile(lo_q)
            hi = non_null[col].quantile(hi_q)
            out[dim] = {
                "high": non_null.filter(pl.col(col) >= hi)["id"].to_list(),
                "low": non_null.filter(pl.col(col) <= lo)["id"].to_list(),
            }
        return out

    # -- pattern mining --------------------------------------------------

    def _behavior_items(self) -> list[_Item]:
        """All Big Five state items, used as candidate behaviors."""
        flat: list[_Item] = []
        for items in self.items_by_dim().values():
            flat.extend(items)
        return flat

    def _situation_columns(self) -> list[tuple[str, str, str]]:
        """Return list of (kind, key, text) for every situation indicator."""
        out: list[tuple[str, str, str]] = []
        for k, t in DIAMONDS_SITUATIONS.items():
            out.append(("diamonds", k, t))
        for k, t in BINARY_SITUATIONS.items():
            out.append(("binary", k, t))
        return out

    @staticmethod
    def _cohens_d(active: list[float], inactive: list[float]) -> float:
        if len(active) < 2 or len(inactive) < 2:
            return 0.0
        m1 = sum(active) / len(active)
        m2 = sum(inactive) / len(inactive)
        v1 = sum((x - m1) ** 2 for x in active) / (len(active) - 1)
        v2 = sum((x - m2) ** 2 for x in inactive) / (len(inactive) - 1)
        n1, n2 = len(active), len(inactive)
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        if pooled <= 0:
            return 0.0
        return (m1 - m2) / math.sqrt(pooled)

    def mine(
        self,
        *,
        min_n_active: int = 30,
        min_abs_d: float = 0.3,
        lo_q: float = 0.25,
        hi_q: float = 0.75,
        within_dim_only: bool = False,
    ) -> list[MinedPattern]:
        """Mine if-then patterns from the dataset.

        Parameters
        ----------
        min_n_active
            Minimum number of observations where the situation is active
            (within the trait-polarity subgroup) to consider the pair.
        min_abs_d
            Minimum |Cohen's d| of the (behavior | situation) coupling.
        lo_q, hi_q
            Trait-polarity quantile cutoffs.
        within_dim_only
            If True, only test behaviors whose Big Five construct equals
            the trait dimension being mined (cleaner, fewer atoms).
            If False (default), all 60 Big Five state items are
            candidate behaviors for every dim x polarity, which is what
            CAPS theory actually predicts -- a high-N person may show
            characteristic *Conscientiousness*-domain behavior in
            specific situations.
        """
        df = self.data
        groups = self.trait_groups(lo_q=lo_q, hi_q=hi_q)
        sit_cols = self._situation_columns()
        beh_items = self._behavior_items()
        patterns: list[MinedPattern] = []

        for dim in BIG_FIVE:
            for polarity in ("high", "low"):
                ids = groups[dim][polarity]
                if len(ids) < 5:
                    continue
                sub = df.filter(pl.col("id").is_in(ids))
                if sub.is_empty():
                    continue

                for sit_kind, sit_key, sit_text in sit_cols:
                    if sit_key not in sub.columns:
                        continue
                    sit_col = pl.col(sit_key).cast(pl.Float64)
                    if sit_kind == "diamonds":
                        # Active = score >= 2 (codebook: 2 = somewhat, 3 = totally)
                        active_mask = sit_col >= 2.0
                        inactive_mask = sit_col == 1.0
                    else:  # binary
                        active_mask = sit_col == 1.0
                        inactive_mask = sit_col == 0.0

                    for beh in beh_items:
                        if within_dim_only and beh.target_dim != dim:
                            continue
                        if beh.name not in sub.columns:
                            continue
                        if beh.name == sit_key:
                            continue

                        b_col = pl.col(beh.name).cast(pl.Float64)
                        if beh.is_reverse:
                            b_col_eff = 6 - b_col
                        else:
                            b_col_eff = b_col

                        active_vals = (
                            sub.filter(active_mask & b_col.is_not_null())
                            .select(b_col_eff.alias("v"))
                            .drop_nulls()["v"]
                            .to_list()
                        )
                        inactive_vals = (
                            sub.filter(inactive_mask & b_col.is_not_null())
                            .select(b_col_eff.alias("v"))
                            .drop_nulls()["v"]
                            .to_list()
                        )
                        if len(active_vals) < min_n_active or len(inactive_vals) < min_n_active:
                            continue

                        d = self._cohens_d(active_vals, inactive_vals)
                        if abs(d) < min_abs_d:
                            continue

                        patterns.append(
                            MinedPattern(
                                target_dimension=dim,
                                polarity=polarity,
                                situation_key=sit_key,
                                situation_text=sit_text,
                                behavior_key=beh.name,
                                behavior_description=beh.description,
                                behavior_construct=beh.construct,
                                behavior_polarity_within_dim=("reverse" if beh.is_reverse else "positive"),
                                effect_size=d,
                                n_situation_active=len(active_vals),
                                n_situation_inactive=len(inactive_vals),
                                mean_active=sum(active_vals) / len(active_vals),
                                mean_inactive=sum(inactive_vals) / len(inactive_vals),
                                n_persons_in_group=len(ids),
                            )
                        )

        return patterns

    # -- contrast filter -------------------------------------------------

    @staticmethod
    def keep_signature_contrasts(
        patterns: Iterable[MinedPattern],
        *,
        min_diff: float = 0.4,
    ) -> list[MinedPattern]:
        """Keep only patterns where the *high* group's effect size for a
        given (dim, situation, behavior) cell differs from the *low*
        group's effect size by ``min_diff``. This is what makes a pattern
        a *trait signature* rather than a universal situation effect.
        """
        by_cell: dict[tuple[str, str, str], dict[str, MinedPattern]] = {}
        for p in patterns:
            cell = (p.target_dimension, p.situation_key, p.behavior_key)
            by_cell.setdefault(cell, {})[p.polarity] = p

        kept: list[MinedPattern] = []
        for cell, by_pol in by_cell.items():
            hi = by_pol.get("high")
            lo = by_pol.get("low")
            if hi is None and lo is None:
                continue
            d_hi = hi.effect_size if hi is not None else 0.0
            d_lo = lo.effect_size if lo is not None else 0.0
            if abs(d_hi - d_lo) < min_diff:
                continue
            # keep whichever pole's effect dominates
            if hi is not None and abs(d_hi) >= abs(d_lo):
                kept.append(hi)
            elif lo is not None:
                kept.append(lo)
        return kept


# ---------------------------------------------------------------------------
# Helpers used by tests + scripts
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(s: str) -> str:
    return _SLUG_RE.sub("_", s.lower()).strip("_")


def pattern_to_atom_id(p: MinedPattern, idx: int) -> str:
    return (
        f"beck2022_{p.polarity}_{p.target_dimension[:4]}__"
        f"{slugify(p.situation_key)[:20]}__"
        f"{slugify(p.behavior_key)[:20]}__{idx:03d}"
    )
