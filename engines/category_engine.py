"""
Category-Aware Engine for AdMIRe 2.0
====================================
Uses knowledge of the 5 image categories to improve ranking:
- Literal: exact words visible (bad apples)
- Literal-related: almost literal (basket of apples)
- Idiomatic: figurative meaning (troublemaker in group)
- Idiomatic-related: close to figurative (dirty person)
- Distractor: superficially related (a peach)

Strategy:
- If LITERAL usage → find images with actual words/objects
- If IDIOMATIC usage → find images with figurative meaning, avoid literal
"""

import time
from typing import List, Tuple, Optional

from .base_engine import BaseEngine, EngineResult
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AdMIReItem
from utils.helpers import (
    parse_ranking_from_response,
    parse_sentence_type_from_response,
    normalize_ranking
)


class CategoryEngine(BaseEngine):
    """
    Category-Aware Engine for idiom image ranking.
    
    Key insight: Each image set contains 5 types of images:
    1. Literal - shows the exact words/objects
    2. Literal-related - almost literal
    3. Idiomatic - shows figurative meaning
    4. Idiomatic-related - close to figurative
    5. Distractor - superficially related but wrong
    
    Strategy:
    - First classify: literal or idiomatic usage
    - Then rank based on category awareness
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "CategoryEngine"
    
    def rank_images(self, item: AdMIReItem) -> EngineResult:
        """
        Rank images using category-aware approach.
        """
        start_time = time.time()
        
        # Step 1: Classify sentence type (reuse proven approach)
        sentence_type, type_confidence = self.classify_sentence_type(
            item.compound, item.sentence
        )
        
        # Step 2: Category-aware ranking
        images_b64 = self._load_images_as_base64(item.image_paths)
        
        if images_b64:
            ranking, raw_response = self._category_ranking_with_images(
                item, sentence_type, images_b64
            )
        else:
            ranking, raw_response = self._category_ranking_captions(
                item, sentence_type
            )
        
        return EngineResult(
            ranking=ranking,
            sentence_type=sentence_type,
            ranking_confidence=0.8,
            sentence_type_confidence=type_confidence,
            engine_name=self.name,
            reasoning=f"Category-aware ranking for {sentence_type} usage",
            raw_response=raw_response,
            processing_time=time.time() - start_time
        )
    
    def classify_sentence_type(
        self, 
        compound: str, 
        sentence: str
    ) -> Tuple[str, float]:
        """
        Classify whether the expression is used literally or idiomatically.
        """
        prompt = f"""Determine if "{compound}" is used LITERALLY or IDIOMATICALLY in this sentence.

SENTENCE: "{sentence}"

LITERAL = The actual, physical objects/words are being described.
- "bad apple" LITERAL = an actual rotten/bad apple fruit
- "green fingers" LITERAL = fingers that are colored green

IDIOMATIC = A figurative/metaphorical meaning, not the actual objects.
- "bad apple" IDIOMATIC = a troublemaker, someone who corrupts others
- "green fingers" IDIOMATIC = skilled at gardening

Look at the context carefully. Is it describing physical objects or a figurative meaning?

Answer with ONE word: LITERAL or IDIOMATIC"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=50,
            model=self.config.api.text_model
        )
        
        return parse_sentence_type_from_response(response)
    
    def _category_ranking_with_images(
        self,
        item: AdMIReItem,
        sentence_type: str,
        images_b64: List[str]
    ) -> Tuple[List[int], str]:
        """Rank using images with category awareness."""
        
        # Split compound into component words for literal matching
        compound_words = item.compound.lower().split()
        
        if sentence_type == "literal":
            prompt = self._build_literal_prompt(item, compound_words)
        else:
            prompt = self._build_idiomatic_prompt(item, compound_words)
        
        response = self._call_llm_with_retry(
            prompt,
            images=images_b64,
            temperature=0.0,
            max_tokens=500
        )
        
        ranking = parse_ranking_from_response(response)
        return normalize_ranking(ranking), response
    
    def _category_ranking_captions(
        self,
        item: AdMIReItem,
        sentence_type: str
    ) -> Tuple[List[int], str]:
        """Rank using captions with category awareness."""
        
        compound_words = item.compound.lower().split()
        captions = item.get_captions()
        
        if sentence_type == "literal":
            prompt = self._build_literal_caption_prompt(item, compound_words, captions)
        else:
            prompt = self._build_idiomatic_caption_prompt(item, compound_words, captions)
        
        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=500,
            model=self.config.api.text_model
        )
        
        ranking = parse_ranking_from_response(response)
        return normalize_ranking(ranking), response
    
    def _build_literal_prompt(self, item: AdMIReItem, compound_words: List[str]) -> str:
        """Build prompt for LITERAL usage - find images with actual objects."""
        
        words_str = " AND ".join([f'"{w}"' for w in compound_words])
        word_list = " and ".join([f'"{w}"' for w in compound_words])
        
        return f"""TASK: Rank images for LITERAL usage of "{item.compound}"

CONTEXT: "{item.sentence}"

The expression "{item.compound}" is used LITERALLY - we want images showing the actual physical objects.

ANALYSIS METHOD for each image:
1. Check: Does the image contain {word_list}?
   - YES (both words visible) → LITERAL (rank highest)
   - PARTIAL (only one word visible) → LITERAL-RELATED (rank second)
   - NO (neither word) → Probably IDIOMATIC/IDIOMATIC-RELATED/DISTRACTOR (rank lower)

2. For images without {word_list}:
   - Does it show something similar to literal but completely different? → DISTRACTOR (rank lowest)
   - Does it show abstract/figurative concepts? → IDIOMATIC/IDIOMATIC-RELATED (rank low)

RANKING STRATEGY:
✓ BEST (Rank 1-2): Images containing {word_list} - these are LITERAL
✓ GOOD (Rank 3): Images with only one word - these are LITERAL-RELATED  
✗ AVOID (Rank 4-5): Images without the words - these are wrong for literal usage

For reference, captions:
{self._format_captions(item.image_captions)}

Analyze each image systematically and provide ranking:
Ranking: N, N, N, N, N"""
    
    def _build_idiomatic_prompt(self, item: AdMIReItem, compound_words: List[str]) -> str:
        """Build prompt for IDIOMATIC usage - find figurative meaning, avoid literal."""
        
        words_str = " or ".join([f'"{w}"' for w in compound_words])
        word_list = " and ".join([f'"{w}"' for w in compound_words])
        
        return f"""TASK: Rank images for IDIOMATIC usage of "{item.compound}"

CONTEXT: "{item.sentence}"

The expression "{item.compound}" is used IDIOMATICALLY - we want images showing the figurative meaning, NOT the literal objects.

ANALYSIS METHOD for each image:

STEP 1: Check for literal objects (AVOID these!)
- Does the image contain {word_list}?
  - YES → LITERAL (WRONG! Rank lowest)
  - PARTIAL (only one word) → LITERAL-RELATED (WRONG! Rank low)
  - NO → Continue to Step 2

STEP 2: For images WITHOUT {word_list}, check idiomatic meaning:
- What is the idiomatic meaning of "{item.compound}"?
- Does the image show this idiomatic meaning?
  - YES, clearly → IDIOMATIC (BEST! Rank highest)
  - CLOSE but not quite → IDIOMATIC-RELATED (Rank second)
  - NO → Continue to Step 3

STEP 3: Check for distractors:
- Does the image show something similar to literal but completely different/irrelevant?
  - YES → DISTRACTOR (Rank lowest)
  - NO → Probably IDIOMATIC-RELATED

RANKING STRATEGY:
✓ BEST (Rank 1-2): Images showing the idiomatic meaning of "{item.compound}" (NO {word_list} visible)
✓ GOOD (Rank 3): Images close to idiomatic meaning
✗ AVOID (Rank 4-5): Images with {word_list} - these are WRONG for idiomatic usage!

For reference, captions:
{self._format_captions(item.image_captions)}

Analyze each image systematically and provide ranking:
Ranking: N, N, N, N, N"""
    
    def _build_literal_caption_prompt(
        self, 
        item: AdMIReItem, 
        compound_words: List[str],
        captions: List[str]
    ) -> str:
        """Build caption-only prompt for LITERAL usage."""
        
        words_str = " AND ".join([f'"{w}"' for w in compound_words])
        word_list = " and ".join([f'"{w}"' for w in compound_words])
        
        return f"""TASK: Rank image descriptions for LITERAL usage of "{item.compound}"

CONTEXT: "{item.sentence}"

The expression is used LITERALLY - we want images showing the actual physical objects.

IMAGE DESCRIPTIONS:
{self._format_captions(captions)}

ANALYSIS METHOD for each description:
1. Check: Does the description mention {word_list}?
   - YES (both words mentioned) → LITERAL (rank highest)
   - PARTIAL (only one word mentioned) → LITERAL-RELATED (rank second)
   - NO (neither word) → Probably IDIOMATIC/IDIOMATIC-RELATED/DISTRACTOR (rank lower)

2. For descriptions without {word_list}:
   - Does it describe something similar to literal but completely different? → DISTRACTOR (rank lowest)
   - Does it describe abstract/figurative concepts? → IDIOMATIC/IDIOMATIC-RELATED (rank low)

RANKING STRATEGY:
✓ BEST (Rank 1-2): Descriptions mentioning {word_list} - these are LITERAL
✓ GOOD (Rank 3): Descriptions with only one word - these are LITERAL-RELATED
✗ AVOID (Rank 4-5): Descriptions without the words - these are wrong for literal usage

Analyze each description systematically:
Ranking: N, N, N, N, N"""
    
    def _build_idiomatic_caption_prompt(
        self, 
        item: AdMIReItem, 
        compound_words: List[str],
        captions: List[str]
    ) -> str:
        """Build caption-only prompt for IDIOMATIC usage."""
        
        words_str = " or ".join([f'"{w}"' for w in compound_words])
        word_list = " and ".join([f'"{w}"' for w in compound_words])
        
        return f"""TASK: Rank image descriptions for IDIOMATIC usage of "{item.compound}"

CONTEXT: "{item.sentence}"

The expression is used IDIOMATICALLY - we want images showing the figurative meaning, NOT the literal objects.

IMAGE DESCRIPTIONS:
{self._format_captions(captions)}

ANALYSIS METHOD for each description:

STEP 1: Check for literal objects (AVOID these!)
- Does the description mention {word_list}?
  - YES → LITERAL (WRONG! Rank lowest)
  - PARTIAL (only one word) → LITERAL-RELATED (WRONG! Rank low)
  - NO → Continue to Step 2

STEP 2: For descriptions WITHOUT {word_list}, check idiomatic meaning:
- What is the idiomatic meaning of "{item.compound}"?
- Does the description suggest an image showing this idiomatic meaning?
  - YES, clearly → IDIOMATIC (BEST! Rank highest)
  - CLOSE but not quite → IDIOMATIC-RELATED (Rank second)
  - NO → Continue to Step 3

STEP 3: Check for distractors:
- Does the description suggest something similar to literal but completely different/irrelevant?
  - YES → DISTRACTOR (Rank lowest)
  - NO → Probably IDIOMATIC-RELATED

RANKING STRATEGY:
✓ BEST (Rank 1-2): Descriptions suggesting the idiomatic meaning of "{item.compound}" (NO {word_list} mentioned)
✓ GOOD (Rank 3): Descriptions close to idiomatic meaning
✗ AVOID (Rank 4-5): Descriptions with {word_list} - these are WRONG for idiomatic usage!

Analyze each description systematically:
Ranking: N, N, N, N, N"""
    
    def _format_captions(self, captions: List[str]) -> str:
        """Format captions for prompt."""
        lines = []
        for i, cap in enumerate(captions, 1):
            cap_short = cap[:150] + "..." if len(cap) > 150 else cap
            lines.append(f"[{i}] {cap_short}")
        return "\n".join(lines)

