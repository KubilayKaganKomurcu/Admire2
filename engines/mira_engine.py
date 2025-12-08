"""
MIRA Engine for AdMIRe 2.0
==========================
Implements the MIRA approach:
- Self-consistency with multiple samples
- Multi-step semantic-visual fusion
- Aggregated ranking from multiple runs

Key insight: Generate multiple rankings with temperature > 0,
then aggregate using weighted voting for robust results.
"""

import time
from typing import List, Tuple, Optional, Dict
from collections import Counter
import numpy as np

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


class MIRAEngine(BaseEngine):
    """
    MIRA: Multimodal Idiom Recognition and Alignment
    
    Training-free framework using:
    1. In-context learning for bias correction
    2. Multi-step semantic-visual fusion
    3. Self-consistency reasoning (multiple samples + aggregation)
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "MIRA"
        self.num_samples = self.config.engine.mira_num_samples
        self.temperature = self.config.engine.mira_temperature
    
    def rank_images(self, item: AdMIReItem) -> EngineResult:
        """
        Rank images using MIRA's multi-step self-consistency approach.
        """
        start_time = time.time()
        
        # Step 1: Classify sentence type (idiomatic vs literal)
        sentence_type, type_confidence = self.classify_sentence_type(
            item.compound, item.sentence
        )
        
        # Step 2: Generate multiple rankings with self-consistency
        all_rankings = []
        all_responses = []
        
        for i in range(self.num_samples):
            ranking, response = self._generate_single_ranking(
                item, sentence_type, temperature=self.temperature
            )
            all_rankings.append(ranking)
            all_responses.append(response)
        
        # Step 3: Aggregate rankings using weighted voting
        final_ranking, ranking_confidence = self._aggregate_rankings(all_rankings)
        
        # Step 4: Calculate per-image scores from aggregation
        image_scores = self._calculate_image_scores(all_rankings)
        
        return EngineResult(
            ranking=final_ranking,
            sentence_type=sentence_type,
            ranking_confidence=ranking_confidence,
            sentence_type_confidence=type_confidence,
            image_scores=image_scores,
            engine_name=self.name,
            reasoning=f"Aggregated from {self.num_samples} samples",
            raw_response="\n---\n".join(all_responses),
            processing_time=time.time() - start_time
        )
    
    def classify_sentence_type(
        self, 
        compound: str, 
        sentence: str
    ) -> Tuple[str, float]:
        """
        Classify whether the expression is used idiomatically or literally.
        Uses zero-shot prompting with explicit instructions.
        """
        prompt = f"""TASK: Determine if "{compound}" is used LITERALLY or IDIOMATICALLY in this sentence.

SENTENCE: "{sentence}"

CRITICAL: Many expressions can be used BOTH ways. You must analyze the CONTEXT carefully:

LITERAL means: The words describe actual, physical, real-world objects/actions.
- Example: "green fingers" is LITERAL if someone's fingers are actually colored green (paint, dye, etc.)
- Example: "couch potato" is LITERAL if describing an actual potato on a couch

IDIOMATIC means: The expression has a figurative/metaphorical meaning different from the words.
- Example: "green fingers" is IDIOMATIC when meaning "good at gardening"
- Example: "couch potato" is IDIOMATIC when meaning "lazy person watching TV"

CONTEXT CLUES TO LOOK FOR:
- Physical actions (dipping, painting, touching) → likely LITERAL
- Abstract descriptions (becoming, being known as) → likely IDIOMATIC
- Mention of actual colors, materials, objects → likely LITERAL
- Mention of skills, behaviors, characteristics → likely IDIOMATIC

NOW ANALYZE: In the sentence "{sentence}", is "{compound}" used:
- To describe something PHYSICAL/REAL? → Answer LITERAL
- To describe something FIGURATIVE/METAPHORICAL? → Answer IDIOMATIC

Your answer (LITERAL or IDIOMATIC):"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=200,
            model=self.config.api.text_model
        )
        
        return parse_sentence_type_from_response(response)
    
    def _generate_single_ranking(
        self, 
        item: AdMIReItem,
        sentence_type: str,
        temperature: float = 0.7
    ) -> Tuple[List[int], str]:
        """Generate a single ranking using multi-step reasoning."""
        
        # Try to load images for multimodal ranking
        images_b64 = self._load_images_as_base64(item.image_paths)
        
        if images_b64:
            # Multimodal ranking with actual images
            return self._multimodal_ranking(item, sentence_type, images_b64, temperature)
        else:
            # Caption-only ranking
            return self._caption_ranking(item, sentence_type, temperature)
    
    def _multimodal_ranking(
        self,
        item: AdMIReItem,
        sentence_type: str,
        images_b64: List[str],
        temperature: float
    ) -> Tuple[List[int], str]:
        """Rank using actual images (multimodal)."""
        
        meaning_context = "idiomatic (figurative)" if sentence_type == "idiomatic" else "literal (word-for-word)"
        
        prompt = f"""You are ranking images for how well they represent a potentially idiomatic expression.

EXPRESSION: "{item.compound}"
SENTENCE: "{item.sentence}"
MEANING TYPE: {meaning_context}

The expression "{item.compound}" is used {meaning_context}ly in this sentence.

I will show you 5 images (Image 1 through Image 5). Your task is to rank them from BEST to WORST based on how well each image depicts the {meaning_context} meaning of "{item.compound}" as used in the sentence.

For reference, here are machine-generated captions for each image:
{self._format_captions(item.image_captions)}

IMPORTANT RANKING GUIDELINES:
- If the meaning is IDIOMATIC: prefer images showing the figurative/metaphorical concept
- If the meaning is LITERAL: prefer images showing the actual physical/literal interpretation
- Consider cultural and contextual relevance
- Images should match the semantic intent, not just contain related objects

After analyzing all images, provide your ranking as a comma-separated list of image numbers from best (first) to worst (last).

OUTPUT FORMAT:
Ranking: X, X, X, X, X

Where X are the image numbers 1-5 in order from best to worst match."""

        response = self._call_llm_with_retry(
            prompt,
            images=images_b64,
            temperature=temperature,
            max_tokens=500
        )
        
        ranking = parse_ranking_from_response(response)
        return normalize_ranking(ranking), response
    
    def _caption_ranking(
        self,
        item: AdMIReItem,
        sentence_type: str,
        temperature: float
    ) -> Tuple[List[int], str]:
        """Rank using captions only (text-based)."""
        
        meaning_context = "idiomatic (figurative)" if sentence_type == "idiomatic" else "literal (word-for-word)"
        captions = item.get_captions()
        
        prompt = f"""TASK: Rank images for how well they represent "{item.compound}" used in its {meaning_context} sense.

SENTENCE: "{item.sentence}"
MEANING: {meaning_context.upper()}

{"IDIOMATIC = figurative meaning (e.g., 'green fingers' = gardening skill, NOT actual green fingers)" if sentence_type == "idiomatic" else "LITERAL = physical/real meaning (e.g., 'green fingers' = fingers actually colored green with paint)"}

IMAGE DESCRIPTIONS:
{self._format_captions(captions)}

CRITICAL: Focus on selecting the TOP 2 images correctly. These matter most!

Step 1: What visual would BEST represent the {meaning_context} meaning?
Step 2: Which description matches that visual?
Step 3: What's the SECOND best match?

For {meaning_context.upper()} "{item.compound}":
{"- Best images show: the CONCEPT/METAPHOR (gardening, laziness, etc.)" if sentence_type == "idiomatic" else "- Best images show: the PHYSICAL/LITERAL thing (actual colors, objects, actions)"}
{"- Avoid images showing: the literal/physical interpretation" if sentence_type == "idiomatic" else "- Avoid images showing: metaphorical/figurative interpretations"}

OUTPUT (5 image numbers, best first):
Ranking:"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=temperature,
            max_tokens=600,
            model=self.config.api.text_model
        )
        
        ranking = parse_ranking_from_response(response)
        return normalize_ranking(ranking), response
    
    def _format_captions(self, captions: List[str]) -> str:
        """Format captions for prompt."""
        lines = []
        for i, cap in enumerate(captions, 1):
            lines.append(f"Image {i}: {cap}")
        return "\n".join(lines)
    
    def _aggregate_rankings(
        self, 
        rankings: List[List[int]]
    ) -> Tuple[List[int], float]:
        """
        Aggregate multiple rankings using Borda count.
        
        Returns:
            Tuple of (aggregated_ranking, confidence_score)
        """
        n_images = 5
        n_rankings = len(rankings)
        
        # Borda count: image at rank 1 gets n_images-1 points, rank 2 gets n_images-2, etc.
        scores = np.zeros(n_images)
        
        for ranking in rankings:
            for position, image_idx in enumerate(ranking):
                if 1 <= image_idx <= n_images:
                    # Points decrease with rank position
                    scores[image_idx - 1] += (n_images - position - 1)
        
        # Convert scores to ranking (higher score = better rank)
        # argsort gives indices that would sort ascending, so we reverse
        sorted_indices = np.argsort(-scores)
        
        # Create ranking where ranking[i] = rank of image i+1
        final_ranking = [0] * n_images
        for rank, img_idx in enumerate(sorted_indices, 1):
            final_ranking[img_idx] = rank
        
        # Calculate confidence based on agreement
        # Higher variance in scores = more agreement = higher confidence
        if scores.std() > 0:
            confidence = min(1.0, scores.std() / (n_images - 1))
        else:
            confidence = 0.5
        
        return final_ranking, confidence
    
    def _calculate_image_scores(self, rankings: List[List[int]]) -> List[float]:
        """Calculate normalized scores for each image from rankings."""
        n_images = 5
        scores = np.zeros(n_images)
        
        for ranking in rankings:
            for position, image_idx in enumerate(ranking):
                if 1 <= image_idx <= n_images:
                    # Convert position to score (position 0 = score 1.0, position 4 = score 0.2)
                    scores[image_idx - 1] += (n_images - position) / n_images
        
        # Normalize
        scores = scores / len(rankings)
        return scores.tolist()
    
    def complete_sequence(self, item: SubtaskBItem) -> Tuple[str, str, float]:
        """Complete image sequence for Subtask B."""
        
        # First classify the sequence type
        sequence_context = f"Image 1: {item.sequence_captions[0]}\nImage 2: {item.sequence_captions[1]}"
        
        prompt = f"""You are analyzing an image sequence related to the expression "{item.compound}".

The sequence shows:
{sequence_context}

This sequence is depicting either:
- IDIOMATIC meaning: the figurative/metaphorical interpretation of "{item.compound}"
- LITERAL meaning: the word-for-word physical interpretation

Which type of meaning does this sequence depict?

Answer: IDIOMATIC or LITERAL"""

        type_response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=100,
            model=self.config.api.text_model
        )
        
        sentence_type, type_conf = parse_sentence_type_from_response(type_response)
        
        # Now select the best completing image
        meaning_desc = "figurative/metaphorical" if sentence_type == "idiomatic" else "literal/physical"
        
        candidates_text = "\n".join([
            f"Option {i+1} ({item.candidate_names[i]}): {cap}"
            for i, cap in enumerate(item.candidate_captions)
        ])
        
        prompt = f"""An image sequence about "{item.compound}" (used in its {meaning_desc} sense):

Sequence so far:
{sequence_context}

Which of these images best completes the sequence?

{candidates_text}

Consider narrative flow and semantic consistency with the {meaning_desc} meaning.

Answer with the option number (1-4):"""

        response = self._call_llm_with_retry(
            prompt,
            temperature=0.0,
            max_tokens=100,
            model=self.config.api.text_model
        )
        
        # Parse the selected option
        import re
        match = re.search(r'[1-4]', response)
        if match:
            selected_idx = int(match.group()) - 1
            selected_image = item.candidate_names[selected_idx]
        else:
            selected_image = item.candidate_names[0]
        
        return selected_image, sentence_type, type_conf

