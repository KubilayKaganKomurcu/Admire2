"""
AdMIRe 2.0 Engines Package
==========================
Contains the three main ranking engines:
- MIRA: Self-consistency + multi-step reasoning
- DAALFT: Detect → Explain → Rank pipeline
- CTYUN-lite: Caption-based zero-shot ranking
"""

from .base_engine import BaseEngine, EngineResult
from .mira_engine import MIRAEngine
from .daalft_engine import DAALFTEngine
from .ctyun_lite_engine import CTYUNLiteEngine

__all__ = [
    "BaseEngine",
    "EngineResult", 
    "MIRAEngine",
    "DAALFTEngine",
    "CTYUNLiteEngine"
]


