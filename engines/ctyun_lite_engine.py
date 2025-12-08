"""
CTYUN-Lite Engine for AdMIRe 2.0
================================
Implements a zero-shot version of the CTYUN approach:
- Caption-based (text-only) ranking
- Direct learning-to-rank without fine-tuning
- Efficient and fast processing

Original CTYUN uses fine-tuning on Qwen. This "lite" version
adapts the core idea for zero-shot inference.

Key insight: Use captions as a proxy for images, enabling
text-only LLMs to participate in the ranking task.
"""

import time
from typing import List, Tuple, Optional

from .base_engine import BaseEngine, EngineResult
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AdMIReItem, SubtaskBItem
from utils.helpers import (
    parse_ranking_from_response,
    parse_sentence_type_from_response,
    normalize_ranking
)


class CTYUNLiteEngine(BaseEngine):
    """
    CTYUN-Lite: Caption-based Text-only Ranking
    
    Zero-shot adaptation of CTYUN approach:
    - Uses image captions instead of pixels
    - Direct ranking without intermediate steps
    - Fast and efficient (text-only)
    
    This is designed to complement the multimodal engines
    by providing a text-only baseline that still performs well.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "CTYUN-Lite"
    
    def rank_images(self, item: AdMIReItem) -> EngineResult:
        """
        Rank images using caption-only approach.
        
        Uses a single, well-crafted prompt that directly outputs ranking.
        """
        start_time = time.time()
        
        # Step 1: Classify sentence type
        sentence_type, type_confidence = self.classify_sentence_type(
            item.compound, item.sentence
        )
        
        # Step 2: Direct ranking from captions
        ranking, image_scores, raw_response = self._direct_ranking(
            item, sentence_type
        )
        
        return EngineResult(
            ranking=ranking,
            sentence_type=sentence_type,
            ranking_confidence=0.7,  # Slightly lower confidence for text-only
            sentence_type_confidence=type_confidence,
            image_scores=image_scores,
            engine_name=self.name,
            reasoning="Direct caption-based ranking",
            raw_response=raw_response,
            processing_time=time.time() - start_time
        )
    
    def classify_sentence_type(
        self, 
        compound: str, 
        sentence: str
    ) -> Tuple[str, float]:
        """
        Classify sentence type using efficient text-only approach.
        """
        prompt = f"""Is "{compound}" used IDIOMATICALLY (figurative meaning) or LITERALLY (word-for-word meaning) in this sentence?

Sentence: "{sentence}"

Answer with just one word: IDIOMATIC or LITERAL"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=50,
            model=self.config.api.text_model
        )
        
        return parse_sentence_type_from_response(response)
    
    def _direct_ranking(
        self,
        item: AdMIReItem,
        sentence_type: str
    ) -> Tuple[List[int], List[float], str]:
        """
        Generate ranking directly from captions.
        
        This mimics CTYUN's direct learning-to-rank approach
        but without fine-tuning.
        """
        captions = item.get_captions()
        meaning_type = "idiomatic (figurative)" if sentence_type == "idiomatic" else "literal"
        
        # Format captions with option shuffling awareness
        # (CTYUN uses option shuffling to remove position bias)
        caption_block = self._format_shuffled_captions(captions)
        
        prompt = f"""Task: Rank image descriptions by relevance to an expression's meaning.

Expression: "{item.compound}"
Context: "{item.sentence}"
Meaning type: {meaning_type}

The expression is used in its {meaning_type} sense. Rank these image descriptions from BEST match (1st) to WORST match (5th):

{caption_block}

Consider:
- Which description best depicts the {meaning_type} meaning of "{item.compound}"?
- Ignore position - evaluate each description on its own merit
- Think about what visual representation would best convey the intended meaning

Output your ranking as 5 numbers separated by commas (image numbers from best to worst):
Ranking:"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=100,
            model=self.config.api.text_model
        )
        
        ranking = parse_ranking_from_response(response)
        ranking = normalize_ranking(ranking)
        
        # Estimate scores from ranking positions
        scores = self._ranking_to_scores(ranking)
        
        return ranking, scores, response
    
    def _format_shuffled_captions(self, captions: List[str]) -> str:
        """
        Format captions with clear numbering.
        
        Note: We don't actually shuffle here (unlike training-time CTYUN)
        because we need consistent numbering for the output.
        """
        lines = []
        for i, cap in enumerate(captions, 1):
            lines.append(f"[{i}] {cap}")
        return "\n".join(lines)
    
    def _ranking_to_scores(self, ranking: List[int]) -> List[float]:
        """
        Convert ranking positions to confidence scores.
        
        Position 1 (best) → 1.0
        Position 5 (worst) → 0.2
        """
        scores = [0.0] * 5
        for img_idx, rank in enumerate(ranking):
            if 1 <= rank <= 5:
                # Convert rank to score: rank 1 → 1.0, rank 5 → 0.2
                scores[img_idx] = (6 - rank) / 5.0
        return scores
    
    def complete_sequence(self, item: SubtaskBItem) -> Tuple[str, str, float]:
        """Complete image sequence for Subtask B (text-only approach)."""
        
        # Classify sequence type
        prompt = f"""Expression: "{item.compound}"

Image sequence:
1. {item.sequence_captions[0]}
2. {item.sequence_captions[1]}

Is this sequence showing the IDIOMATIC or LITERAL meaning of "{item.compound}"?

Answer: IDIOMATIC or LITERAL"""

        type_response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=50,
            model=self.config.api.text_model
        )
        
        sentence_type, type_conf = parse_sentence_type_from_response(type_response)
        
        # Select completing image
        meaning_desc = "figurative" if sentence_type == "idiomatic" else "literal"
        
        candidates = "\n".join([
            f"[{i+1}] {cap}" for i, cap in enumerate(item.candidate_captions)
        ])
        
        prompt = f"""Expression: "{item.compound}" (used in {meaning_desc} sense)

Current sequence:
1. {item.sequence_captions[0]}
2. {item.sequence_captions[1]}

Which image best completes this sequence?
{candidates}

Answer with just the number (1-4):"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=20,
            model=self.config.api.text_model
        )
        
        import re
        match = re.search(r'[1-4]', response)
        selected_idx = int(match.group()) - 1 if match else 0
        
        return item.candidate_names[selected_idx], sentence_type, type_conf
    
    def batch_rank(self, items: List[AdMIReItem]) -> List[EngineResult]:
        """
        Efficiently rank multiple items.
        
        CTYUN-Lite is fast enough for batch processing.
        """
        results = []
        for item in items:
            result = self.rank_images(item)
            results.append(result)
        return results

