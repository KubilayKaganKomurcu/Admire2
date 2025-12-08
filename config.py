"""
AdMIRe 2.0 Configuration
========================
Central configuration for the ensemble system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class APIConfig:
    """API configuration for LLM providers."""
    # API key loaded from environment variable for security
    # Set with: export OPENAI_API_KEY="your-key" (Linux/Mac)
    # Or: $env:OPENAI_API_KEY="your-key" (PowerShell)
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_base_url: str = "https://api.openai.com/v1"
    
    # Model choices (GPT-5 series - better AND cheaper than GPT-4!)
    # Options: gpt-5, gpt-5-mini, gpt-5-nano
    # Pricing: gpt-5 ($1.25/$10), gpt-5-mini ($0.25/$2), gpt-5-nano ($0.05/$0.40) per 1M tokens
    vision_model: str = "gpt-5"  # For image + text (best quality)
    text_model: str = "gpt-5-mini"  # For text-only (5x cheaper, still excellent)
    reasoning_model: str = "gpt-5"  # For complex reasoning
    
    # Alternative: Use nano for maximum cost savings (lower quality)
    # vision_model: str = "gpt-5-mini"
    # text_model: str = "gpt-5-nano"
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class DataConfig:
    """Data paths configuration."""
    base_path: str = "data"
    
    # Subtask A paths
    subtask_a_english_path: str = "data/subtask_a/english"
    subtask_a_portuguese_path: str = "data/subtask_a/portuguese"
    
    # Subtask B path
    subtask_b_path: str = "data/subtask_b"
    
    # File names
    train_file: str = "subtask_a_train.tsv"
    dev_file: str = "subtask_a_dev.tsv"
    test_file: str = "subtask_a_test.tsv"


@dataclass 
class EngineConfig:
    """Configuration for individual engines."""
    
    # MIRA settings
    mira_num_samples: int = 3  # Number of samples for self-consistency
    mira_temperature: float = 0.7  # Temperature for sampling diversity
    
    # DAALFT settings
    daalft_include_explanation: bool = True  # Whether to generate explanations
    daalft_chain_of_thought: bool = True  # Use CoT prompting
    
    # CTYUN-lite settings
    ctyun_use_captions_only: bool = True  # Text-only mode
    ctyun_caption_augmentation: bool = False  # Augment captions (disabled for zero-shot)


@dataclass
class EnsembleConfig:
    """Ensemble aggregation settings."""
    
    # Which engines to use
    enabled_engines: List[str] = field(default_factory=lambda: ["mira", "daalft", "ctyun_lite"])
    
    # Aggregation method: "borda", "weighted_borda", "voting", "mean_rank"
    aggregation_method: str = "weighted_borda"
    
    # Engine weights (used if aggregation_method is weighted)
    engine_weights: dict = field(default_factory=lambda: {
        "mira": 1.0,
        "daalft": 1.0,
        "ctyun_lite": 0.8  # Slightly lower weight for text-only
    })
    
    # Confidence threshold for sentence type classification
    sentence_type_threshold: float = 0.6


@dataclass
class AdMIRe2Config:
    """Master configuration combining all settings."""
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    
    # General settings
    verbose: bool = True
    save_intermediate: bool = True
    output_dir: str = "outputs"
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)


# Default configuration instance
CONFIG = AdMIRe2Config()


def get_config() -> AdMIRe2Config:
    """Get the default configuration."""
    return CONFIG


def update_config(**kwargs) -> AdMIRe2Config:
    """Update configuration with custom values."""
    global CONFIG
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
    return CONFIG

