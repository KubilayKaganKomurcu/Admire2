"""
AdMIRe 2.0 Base Engine
======================
Abstract base class for all ranking engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AdMIReItem, SubtaskBItem
from config import AdMIRe2Config, get_config


@dataclass
class EngineResult:
    """Result from a ranking engine."""
    
    # Core outputs
    ranking: List[int]  # 1-indexed ranking [1,2,3,4,5] where position i = rank of image i
    sentence_type: str  # "idiomatic" or "literal"
    
    # Confidence scores
    ranking_confidence: float = 0.0
    sentence_type_confidence: float = 0.0
    
    # Per-image scores (optional)
    image_scores: Optional[List[float]] = None
    
    # Metadata
    engine_name: str = ""
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    
    # Timing
    processing_time: float = 0.0
    
    def to_ordered_images(self, image_names: List[str]) -> List[str]:
        """Convert ranking to ordered list of image names."""
        if len(self.ranking) != len(image_names):
            return image_names
        
        # ranking[i-1] = rank of image i (1-indexed)
        # We want to return images sorted by their rank
        paired = list(zip(self.ranking, image_names))
        paired.sort(key=lambda x: x[0])
        return [name for _, name in paired]


class BaseEngine(ABC):
    """
    Abstract base class for AdMIRe ranking engines.
    
    All engines must implement:
    - rank_images(): Main ranking method for Subtask A
    - classify_sentence_type(): Classify idiomatic vs literal
    
    Optional:
    - complete_sequence(): For Subtask B
    """
    
    def __init__(self, config: Optional[AdMIRe2Config] = None):
        self.config = config or get_config()
        self.client = OpenAI(
            api_key=self.config.api.openai_api_key,
            base_url=self.config.api.openai_base_url,
            timeout=120.0  # 2 minute timeout for image processing
        )
        self.name = self.__class__.__name__
    
    @abstractmethod
    def rank_images(self, item: AdMIReItem) -> EngineResult:
        """
        Rank the 5 candidate images for a given item.
        
        Args:
            item: AdMIReItem containing compound, sentence, and image info
        
        Returns:
            EngineResult with ranking and sentence type
        """
        pass
    
    @abstractmethod
    def classify_sentence_type(
        self, 
        compound: str, 
        sentence: str
    ) -> Tuple[str, float]:
        """
        Classify whether the compound is used idiomatically or literally.
        
        Args:
            compound: The potentially idiomatic expression
            sentence: Context sentence containing the compound
        
        Returns:
            Tuple of (sentence_type, confidence)
        """
        pass
    
    def complete_sequence(self, item: SubtaskBItem) -> Tuple[str, str, float]:
        """
        Complete image sequence for Subtask B.
        
        Args:
            item: SubtaskBItem with sequence and candidate info
        
        Returns:
            Tuple of (selected_image_name, sentence_type, confidence)
        """
        # Default implementation - can be overridden
        raise NotImplementedError(
            f"{self.name} does not implement Subtask B sequence completion"
        )
    
    def _call_llm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        model: Optional[str] = None
    ) -> str:
        """
        Make a call to the LLM.
        
        Args:
            prompt: Text prompt
            images: Optional list of base64-encoded images
            temperature: Sampling temperature (GPT-5 only supports 1.0)
            max_tokens: Maximum response tokens
            model: Model to use (defaults to config)
        
        Returns:
            LLM response text
        """
        model = model or self.config.api.vision_model
        
        messages = []
        
        if images:
            # Multimodal message with images
            content = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "auto"  # Let model decide detail level
                    }
                })
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": prompt})
        
        # GPT-5 models only support temperature=1, so we don't pass it
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens
            )
            
            # Debug: check response structure
            if not response.choices:
                print(f"  [API DEBUG] No choices in response!")
                return ""
            
            content = response.choices[0].message.content
            if content is None:
                print(f"  [API DEBUG] Response content is None! Finish reason: {response.choices[0].finish_reason}")
                return ""
            
            return content
        except Exception as e:
            print(f"  [API DEBUG] API call failed: {type(e).__name__}: {e}")
            raise
    
    def _call_llm_with_retry(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        max_retries: int = 10,
        unlimited: bool = True
    ) -> str:
        """Call LLM with retry logic.
        
        Args:
            unlimited: If True, retry forever until success
        """
        import time
        
        attempt = 0
        while True:
            try:
                result = self._call_llm(prompt, images, temperature, max_tokens, model)
                # Check for empty response and retry
                if not result or len(result.strip()) == 0:
                    print(f"  [RETRY] Empty response on attempt {attempt + 1}, retrying...")
                    attempt += 1
                    wait_time = min(30, 2 ** min(attempt, 5))  # Cap at 30 seconds
                    time.sleep(wait_time)
                    if not unlimited and attempt >= max_retries:
                        print(f"  [RETRY] Max retries ({max_retries}) reached with empty responses")
                        return ""
                    continue
                return result
            except Exception as e:
                attempt += 1
                print(f"  [RETRY] Attempt {attempt} failed: {type(e).__name__}: {e}")
                
                if not unlimited and attempt >= max_retries:
                    raise
                
                wait_time = min(30, 2 ** min(attempt, 5))  # Cap at 30 seconds
                print(f"  [RETRY] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    def _load_images_as_base64(self, image_paths: List[str]) -> List[str]:
        """Load images from paths and encode as base64."""
        from utils.helpers import encode_image_to_base64
        
        encoded = []
        total_size = 0
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                # Check file size
                file_size = os.path.getsize(path)
                total_size += file_size
                size_kb = file_size / 1024
                if size_kb > 500:
                    print(f"  [IMAGE] Warning: Image {i+1} is large ({size_kb:.1f}KB): {path}")
                
                encoded.append(encode_image_to_base64(path))
            else:
                # Return empty list if any image is missing
                print(f"  [IMAGE] Missing: {path}")
                return []
        
        print(f"  [IMAGE] Loaded {len(encoded)} images, total size: {total_size/1024:.1f}KB")
        return encoded

