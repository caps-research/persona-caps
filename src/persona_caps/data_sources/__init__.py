"""Empirical data sources for paper-track if-then atom mining.

Each submodule implements the same conceptual interface:

* ``load()``   -- fetch / cache the raw dataset
* ``mine()``   -- run the statistical mining pipeline that turns ESM
  observations into ``MinedPattern`` records (one per surviving
  situation x behavior x trait-polarity combination)
* ``verbalize()`` -- (optional) LLM step that turns ``MinedPattern``
  records into natural-language ``IfThenAtom`` records

Provenance is mandatory on every mined atom (CONSENSUS §2 D2/D3).
"""

from .beck2022 import (
    Beck2022,
    MinedPattern,
)
from .ipip_neo import IpipItem, load_items as load_ipip_items

__all__ = ["Beck2022", "MinedPattern", "IpipItem", "load_ipip_items"]
