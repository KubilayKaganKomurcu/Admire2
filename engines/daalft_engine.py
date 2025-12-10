"""
DAALFT Engine for AdMIRe 2.0
============================
Implements the DAALFT approach:
- Multi-step zero-shot reasoning pipeline
- Explicit idiomaticity detection
- Optional explanation generation
- Structured: Detect → Explain → Rank

Key insight: Force explicit reasoning at each step for
better interpretability and accuracy.
"""

import time
from typing import List, Tuple, Optional, Dict

from .base_engine import BaseEngine, EngineResult
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AdMIReItem, SubtaskBItem
from utils.helpers import (
    parse_ranking_from_response,
    parse_sentence_type_from_response,
    normalize_ranking,
    parse_json_from_response
)


class DAALFTEngine(BaseEngine):
    """
    DAALFT: Detect-Analyze-Align-Rank with Language and Text Features
    
    Training-free pipeline with explicit reasoning steps:
    1. DETECT: Identify if expression is idiomatic or literal
    2. EXPLAIN: Generate explanation of the intended meaning
    3. RANK: Score images based on semantic alignment
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "DAALFT"
        self.include_explanation = self.config.engine.daalft_include_explanation
        self.use_cot = self.config.engine.daalft_chain_of_thought
    
    def rank_images(self, item: AdMIReItem) -> EngineResult:
        """
        Rank images using DAALFT's detect → explain → rank pipeline.
        """
        start_time = time.time()
        reasoning_steps = []
        
        # Step 1: DETECT - Classify sentence type
        sentence_type, type_confidence, detect_reasoning = self._detect_idiomaticity(
            item.compound, item.sentence
        )
        reasoning_steps.append(f"DETECT: {detect_reasoning}")
        
        # Step 2: EXPLAIN - Generate meaning explanation
        if self.include_explanation:
            explanation = self._explain_meaning(
                item.compound, item.sentence, sentence_type
            )
            reasoning_steps.append(f"EXPLAIN: {explanation}")
        else:
            explanation = None
        
        # Step 3: RANK - Score and rank images
        ranking, image_scores, rank_reasoning = self._rank_images(
            item, sentence_type, explanation
        )
        reasoning_steps.append(f"RANK: {rank_reasoning}")
        
        return EngineResult(
            ranking=ranking,
            sentence_type=sentence_type,
            ranking_confidence=0.8,  # DAALFT is generally confident
            sentence_type_confidence=type_confidence,
            image_scores=image_scores,
            engine_name=self.name,
            reasoning="\n\n".join(reasoning_steps),
            processing_time=time.time() - start_time
        )
    
    def classify_sentence_type(
        self, 
        compound: str, 
        sentence: str
    ) -> Tuple[str, float]:
        """Public interface for sentence type classification."""
        sentence_type, confidence, _ = self._detect_idiomaticity(compound, sentence)
        return sentence_type, confidence
    
    def _detect_idiomaticity(
        self, 
        compound: str, 
        sentence: str
    ) -> Tuple[str, float, str]:
        """
        Step 1: Detect whether expression is idiomatic or literal.
        
        Returns:
            Tuple of (sentence_type, confidence, reasoning)
        """
        prompt = f"""TASK: Idiomaticity Detection - Classify LITERAL vs IDIOMATIC

SENTENCE: "{sentence}"
EXPRESSION: "{compound}"

IMPORTANT: Even well-known idioms can be used LITERALLY in certain contexts!

LITERAL = The words describe actual, physical, real things:
- "green fingers" → fingers that are actually green-colored (paint, dye)
- "couch potato" → an actual potato sitting on a couch
- Look for: physical actions, colors, materials, concrete objects

IDIOMATIC = Figurative/metaphorical meaning:
- "green fingers" → skilled at gardening
- "couch potato" → lazy person who watches TV
- Look for: descriptions of skills, personality, abstract qualities

CONTEXT ANALYSIS for "{sentence}":
- Are there physical actions (dipping, painting, holding)? → LITERAL
- Are there mentions of actual colors/materials? → LITERAL  
- Is it describing a skill or behavior? → IDIOMATIC
- Is it describing someone's character? → IDIOMATIC

OUTPUT FORMAT (JSON):
{{
    "classification": "LITERAL" or "IDIOMATIC",
    "confidence": 0.0 to 1.0,
    "key_evidence": "the words in the sentence that determined your choice",
    "reasoning": "brief explanation"
}}

JSON response:"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=400,
            model=self.config.api.text_model
        )
        
        # Parse response
        parsed = parse_json_from_response(response)
        
        if parsed:
            classification = parsed.get("classification", "").upper()
            sentence_type = "idiomatic" if "IDIOMATIC" in classification else "literal"
            confidence = float(parsed.get("confidence", 0.7))
            reasoning = parsed.get("reasoning", "")
        else:
            # Fallback parsing
            sentence_type, confidence = parse_sentence_type_from_response(response)
            reasoning = response[:200]
        
        return sentence_type, confidence, reasoning
    
    def _explain_meaning(
        self, 
        compound: str, 
        sentence: str,
        sentence_type: str
    ) -> str:
        """
        Step 2: Generate explanation of the intended meaning.
        
        This explanation is used to guide image ranking.
        """
        meaning_type = "idiomatic (figurative)" if sentence_type == "idiomatic" else "literal"
        
        prompt = f"""TASK: Meaning Explanation

The expression "{compound}" is used {meaning_type}ly in this sentence:
"{sentence}"

Generate a detailed explanation of what "{compound}" means in this context, focusing on:
1. The specific semantic content being conveyed
2. What visual elements would best represent this meaning
3. What someone would expect to see in an image depicting this meaning

EXPLANATION (2-3 sentences):"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=200,
            model=self.config.api.text_model
        )
        
        return response.strip()
    
    def _rank_images(
        self,
        item: AdMIReItem,
        sentence_type: str,
        explanation: Optional[str]
    ) -> Tuple[List[int], List[float], str]:
        """
        Step 3: Rank images based on semantic alignment.
        
        Returns:
            Tuple of (ranking, image_scores, reasoning)
        """
        # Try multimodal ranking first
        images_b64 = self._load_images_as_base64(item.image_paths)
        
        if images_b64:
            return self._rank_with_images(item, sentence_type, explanation, images_b64)
        else:
            return self._rank_with_captions(item, sentence_type, explanation)
    
    def _rank_with_images(
        self,
        item: AdMIReItem,
        sentence_type: str,
        explanation: Optional[str],
        images_b64: List[str]
    ) -> Tuple[List[int], List[float], str]:
        """Rank using actual images."""
        
        meaning_desc = "figurative/metaphorical" if sentence_type == "idiomatic" else "literal/physical"
        
        explanation_block = ""
        if explanation:
            explanation_block = f"\nMEANING EXPLANATION:\n{explanation}\n"
        
        prompt = f"""TASK: Image Ranking for Idiom Understanding

EXPRESSION: "{item.compound}"
SENTENCE: "{item.sentence}"
MEANING TYPE: {sentence_type.upper()} ({meaning_desc})
{explanation_block}
You will see 5 images. For reference, here are their auto-generated captions:
{self._format_captions(item.image_captions)}

SCORING CRITERIA:
- How well does each image visually represent the {meaning_desc} meaning?
- Does the image capture the semantic essence of "{item.compound}" as used here?
- Consider cultural context and visual metaphors.

TASK: Assign a score from 1-10 to each image, then provide a ranking from best to worst.

OUTPUT FORMAT (JSON):
{{
    "scores": [score1, score2, score3, score4, score5],
    "ranking": [best_img, ..., worst_img],
    "reasoning": "brief explanation"
}}

Respond with JSON only:"""

        response = self._call_llm_with_retry(
            prompt,
            images=images_b64,
            temperature=0.0,
            max_tokens=400
        )
        
        return self._parse_ranking_response(response)
    
    def _rank_with_captions(
        self,
        item: AdMIReItem,
        sentence_type: str,
        explanation: Optional[str]
    ) -> Tuple[List[int], List[float], str]:
        """Rank using captions only."""
        
        meaning_desc = "figurative/metaphorical" if sentence_type == "idiomatic" else "literal/physical"
        captions = item.get_captions()
        
        explanation_block = ""
        if explanation:
            explanation_block = f"\nMEANING EXPLANATION:\n{explanation}\n"
        
        prompt = f"""EXPRESSION: "{item.compound}"
SENTENCE: "{item.sentence}"
MEANING TYPE: {sentence_type.upper()} ({meaning_desc})
{explanation_block}
IMAGE DESCRIPTIONS:
{self._format_captions(captions)}

Rank these 5 images from BEST to WORST for depicting the {meaning_desc} meaning.

CRITICAL: Respond with ONLY valid JSON in this exact format:
{{"scores": [7, 5, 8, 3, 6], "ranking": [3, 1, 2, 5, 4]}}

The "ranking" array must contain numbers 1-5 (image numbers) ordered from best to worst.

JSON response:"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=400,
            model=self.config.api.text_model
        )
        
        return self._parse_ranking_response(response)
    
    def _format_captions(self, captions: List[str]) -> str:
        """Format captions for prompt."""
        lines = []
        for i, cap in enumerate(captions, 1):
            lines.append(f"Image {i}: {cap}")
        return "\n".join(lines)
    
    def _parse_ranking_response(
        self, 
        response: str
    ) -> Tuple[List[int], List[float], str]:
        """Parse the ranking response from LLM."""
        
        parsed = parse_json_from_response(response)
        
        if parsed:
            # Get scores (normalize to 0-1)
            raw_scores = parsed.get("scores", [5, 5, 5, 5, 5])
            scores = [s / 10.0 for s in raw_scores[:5]]
            while len(scores) < 5:
                scores.append(0.5)
            
            # Get ranking
            ranking = parsed.get("ranking", [1, 2, 3, 4, 5])
            ranking = normalize_ranking(ranking[:5])
            
            reasoning = parsed.get("reasoning", "")
        else:
            # Fallback parsing
            ranking = parse_ranking_from_response(response)
            ranking = normalize_ranking(ranking)
            scores = [0.5] * 5
            reasoning = response[:200]
        
        return ranking, scores, reasoning
    
    def complete_sequence(self, item: SubtaskBItem) -> Tuple[str, str, float]:
        """Complete image sequence for Subtask B using DAALFT approach."""
        
        # Step 1: Detect meaning type from sequence
        sequence_desc = f"""Image 1: {item.sequence_captions[0]}
Image 2: {item.sequence_captions[1]}"""
        
        prompt = f"""TASK: Sequence Meaning Detection

Expression: "{item.compound}"
Image sequence:
{sequence_desc}

Is this sequence depicting the IDIOMATIC (figurative) or LITERAL meaning of "{item.compound}"?

OUTPUT FORMAT (JSON):
{{
    "meaning_type": "IDIOMATIC" or "LITERAL",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Respond with JSON only:"""

        type_response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=200,
            model=self.config.api.text_model
        )
        
        type_parsed = parse_json_from_response(type_response)
        if type_parsed:
            sentence_type = "idiomatic" if "IDIOMATIC" in type_parsed.get("meaning_type", "").upper() else "literal"
            type_conf = float(type_parsed.get("confidence", 0.7))
        else:
            sentence_type, type_conf = parse_sentence_type_from_response(type_response)
        
        # Step 2: Select completing image
        meaning_desc = "figurative/metaphorical" if sentence_type == "idiomatic" else "literal/physical"
        
        candidates = "\n".join([
            f"Option {i+1}: {cap}" for i, cap in enumerate(item.candidate_captions)
        ])
        
        prompt = f"""TASK: Sequence Completion

Expression: "{item.compound}" (used in {meaning_desc} sense)
Current sequence:
{sequence_desc}

Candidate images to complete the sequence:
{candidates}

Which image best completes this sequence while maintaining the {meaning_desc} meaning?

OUTPUT FORMAT (JSON):
{{
    "selected_option": 1-4,
    "reasoning": "brief explanation"
}}

Respond with JSON only:"""

        select_response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=200,
            model=self.config.api.text_model
        )
        
        select_parsed = parse_json_from_response(select_response)
        if select_parsed:
            selected_idx = int(select_parsed.get("selected_option", 1)) - 1
            selected_idx = max(0, min(3, selected_idx))
        else:
            # Fallback
            import re
            match = re.search(r'[1-4]', select_response)
            selected_idx = int(match.group()) - 1 if match else 0
        
        return item.candidate_names[selected_idx], sentence_type, type_conf

