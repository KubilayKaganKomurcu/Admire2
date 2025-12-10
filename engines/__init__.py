"""
AdMIRe 2.0 Engines Package
==========================
Contains the ranking engines:
- MIRA: Self-consistency + multi-step reasoning
- DAALFT: Detect → Explain → Rank pipeline
- CTYUN-lite: Caption-based zero-shot ranking
- CategoryEngine: Category-aware ranking (literal vs idiomatic)
"""

from .base_engine import BaseEngine, EngineResult
from .mira_engine import MIRAEngine
from .daalft_engine import DAALFTEngine
from .ctyun_lite_engine import CTYUNLiteEngine
from .category_engine import CategoryEngine

__all__ = [
    "BaseEngine",
    "EngineResult", 
    "MIRAEngine",
    "DAALFTEngine",
    "CTYUNLiteEngine",
    "CategoryEngine"
]


