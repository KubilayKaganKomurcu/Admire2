"""
AdMIRe 2.0 Ensemble Aggregator
==============================
Combines predictions from multiple engines for robust ranking.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.base_engine import EngineResult
from config import EnsembleConfig, get_config


@dataclass
class EnsembleResult:
    """Result from ensemble aggregation."""
    ranking: List[int]
    sentence_type: str
    ranking_confidence: float
    sentence_type_confidence: float
    
    # Per-engine results
    engine_results: Dict[str, EngineResult]
    
    # Aggregation metadata
    aggregation_method: str
    agreement_score: float  # How much engines agreed
    
    def to_ordered_images(self, image_names: List[str]) -> List[str]:
        """Convert ranking to ordered list of image names."""
        if len(self.ranking) != len(image_names):
            return image_names
        
        paired = list(zip(self.ranking, image_names))
        paired.sort(key=lambda x: x[0])
        return [name for _, name in paired]


class EnsembleAggregator:
    """
    Aggregates predictions from multiple engines.
    
    Supports multiple aggregation methods:
    - Borda count: Points based on ranking positions
    - Weighted Borda: Borda with engine-specific weights
    - Mean rank: Average rank across engines
    - Voting: Majority vote for each position
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or get_config().ensemble
        self.method = self.config.aggregation_method
        self.weights = self.config.engine_weights
    
    def aggregate(
        self, 
        engine_results: Dict[str, EngineResult]
    ) -> EnsembleResult:
        """
        Aggregate multiple engine results into a single prediction.
        
        Args:
            engine_results: Dict mapping engine name to its result
        
        Returns:
            EnsembleResult with aggregated prediction
        """
        if not engine_results:
            raise ValueError("No engine results to aggregate")
        
        # Aggregate rankings
        if self.method == "borda":
            ranking, confidence = self._borda_aggregation(engine_results)
        elif self.method == "weighted_borda":
            ranking, confidence = self._weighted_borda_aggregation(engine_results)
        elif self.method == "mean_rank":
            ranking, confidence = self._mean_rank_aggregation(engine_results)
        elif self.method == "voting":
            ranking, confidence = self._voting_aggregation(engine_results)
        else:
            ranking, confidence = self._weighted_borda_aggregation(engine_results)
        
        # Aggregate sentence type
        sentence_type, type_confidence = self._aggregate_sentence_type(engine_results)
        
        # Calculate agreement score
        agreement = self._calculate_agreement(engine_results)
        
        return EnsembleResult(
            ranking=ranking,
            sentence_type=sentence_type,
            ranking_confidence=confidence,
            sentence_type_confidence=type_confidence,
            engine_results=engine_results,
            aggregation_method=self.method,
            agreement_score=agreement
        )
    
    def _borda_aggregation(
        self, 
        results: Dict[str, EngineResult]
    ) -> Tuple[List[int], float]:
        """
        Borda count aggregation.
        
        Each engine gives points: rank 1 gets 4 points, rank 2 gets 3, etc.
        """
        n_images = 5
        scores = np.zeros(n_images)
        
        for result in results.values():
            ranking = result.ranking
            for img_idx, rank in enumerate(ranking):
                if 1 <= rank <= n_images:
                    # Points = n_images - rank (higher rank = more points)
                    scores[img_idx] += (n_images - rank)
        
        # Convert scores to ranking
        final_ranking = self._scores_to_ranking(scores)
        
        # Confidence based on score separation
        confidence = self._calculate_confidence_from_scores(scores)
        
        return final_ranking, confidence
    
    def _weighted_borda_aggregation(
        self, 
        results: Dict[str, EngineResult]
    ) -> Tuple[List[int], float]:
        """
        Weighted Borda count - engines have different weights.
        """
        n_images = 5
        scores = np.zeros(n_images)
        total_weight = 0
        
        for engine_name, result in results.items():
            weight = self.weights.get(engine_name.lower().replace("-", "_"), 1.0)
            total_weight += weight
            
            ranking = result.ranking
            for img_idx, rank in enumerate(ranking):
                if 1 <= rank <= n_images:
                    scores[img_idx] += weight * (n_images - rank)
        
        # Normalize by total weight
        if total_weight > 0:
            scores = scores / total_weight
        
        final_ranking = self._scores_to_ranking(scores)
        confidence = self._calculate_confidence_from_scores(scores)
        
        return final_ranking, confidence
    
    def _mean_rank_aggregation(
        self, 
        results: Dict[str, EngineResult]
    ) -> Tuple[List[int], float]:
        """
        Mean rank aggregation - average rank for each image.
        """
        n_images = 5
        rank_sums = np.zeros(n_images)
        
        for result in results.values():
            ranking = result.ranking
            for img_idx, rank in enumerate(ranking):
                rank_sums[img_idx] += rank
        
        # Average ranks
        avg_ranks = rank_sums / len(results)
        
        # Convert to final ranking (lower avg rank = better position)
        sorted_indices = np.argsort(avg_ranks)
        final_ranking = [0] * n_images
        for position, img_idx in enumerate(sorted_indices, 1):
            final_ranking[img_idx] = position
        
        # Confidence from rank variance
        rank_variance = np.var([r.ranking for r in results.values()], axis=0).mean()
        confidence = max(0.3, 1.0 - rank_variance / 4)
        
        return final_ranking, confidence
    
    def _voting_aggregation(
        self, 
        results: Dict[str, EngineResult]
    ) -> Tuple[List[int], float]:
        """
        Voting aggregation - most common rank for each image.
        """
        n_images = 5
        final_ranking = []
        
        for img_idx in range(n_images):
            ranks_for_image = [r.ranking[img_idx] for r in results.values()]
            # Most common rank
            most_common = Counter(ranks_for_image).most_common(1)[0][0]
            final_ranking.append(most_common)
        
        # Handle ties and ensure valid ranking
        final_ranking = self._ensure_valid_ranking(final_ranking)
        
        # Confidence from voting agreement
        agreements = 0
        for img_idx in range(n_images):
            ranks = [r.ranking[img_idx] for r in results.values()]
            agreements += max(Counter(ranks).values()) / len(ranks)
        confidence = agreements / n_images
        
        return final_ranking, confidence
    
    def _scores_to_ranking(self, scores: np.ndarray) -> List[int]:
        """Convert score array to ranking list."""
        n_images = len(scores)
        sorted_indices = np.argsort(-scores)  # Descending order
        
        ranking = [0] * n_images
        for position, img_idx in enumerate(sorted_indices, 1):
            ranking[img_idx] = position
        
        return ranking
    
    def _ensure_valid_ranking(self, ranking: List[int]) -> List[int]:
        """Ensure ranking is a valid permutation of [1,2,3,4,5]."""
        n = len(ranking)
        
        # If already valid, return as-is
        if set(ranking) == set(range(1, n + 1)):
            return ranking
        
        # Otherwise, fix by assigning missing ranks
        used = set()
        result = [0] * n
        
        # First pass: keep valid unique ranks
        for i, r in enumerate(ranking):
            if 1 <= r <= n and r not in used:
                result[i] = r
                used.add(r)
        
        # Second pass: assign missing ranks
        missing = [r for r in range(1, n + 1) if r not in used]
        missing_idx = 0
        for i in range(n):
            if result[i] == 0:
                result[i] = missing[missing_idx]
                missing_idx += 1
        
        return result
    
    def _aggregate_sentence_type(
        self, 
        results: Dict[str, EngineResult]
    ) -> Tuple[str, float]:
        """Aggregate sentence type predictions."""
        types = []
        confidences = []
        
        for engine_name, result in results.items():
            weight = self.weights.get(engine_name.lower().replace("-", "_"), 1.0)
            types.append((result.sentence_type, weight, result.sentence_type_confidence))
        
        # Weighted voting
        idiomatic_score = sum(w * c for t, w, c in types if t == "idiomatic")
        literal_score = sum(w * c for t, w, c in types if t == "literal")
        
        if idiomatic_score > literal_score:
            sentence_type = "idiomatic"
            confidence = idiomatic_score / (idiomatic_score + literal_score + 1e-6)
        else:
            sentence_type = "literal"
            confidence = literal_score / (idiomatic_score + literal_score + 1e-6)
        
        return sentence_type, confidence
    
    def _calculate_agreement(self, results: Dict[str, EngineResult]) -> float:
        """Calculate how much engines agree on the ranking."""
        if len(results) < 2:
            return 1.0
        
        rankings = [r.ranking for r in results.values()]
        
        # Calculate pairwise agreement
        total_agreement = 0
        n_pairs = 0
        
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                # Count matching positions
                matches = sum(1 for a, b in zip(rankings[i], rankings[j]) if a == b)
                total_agreement += matches / 5
                n_pairs += 1
        
        return total_agreement / n_pairs if n_pairs > 0 else 1.0
    
    def _calculate_confidence_from_scores(self, scores: np.ndarray) -> float:
        """Calculate confidence based on score distribution."""
        if len(scores) < 2:
            return 0.5
        
        # Higher variance = more decisive ranking = higher confidence
        score_range = scores.max() - scores.min()
        max_range = len(scores) * (len(scores) - 1) / 2  # Max possible Borda difference
        
        return min(1.0, 0.5 + score_range / (2 * max_range))


def create_ensemble_from_engines(
    engines: List,
    item,
    config: Optional[EnsembleConfig] = None
) -> EnsembleResult:
    """
    Convenience function to run multiple engines and aggregate results.
    
    Args:
        engines: List of engine instances
        item: AdMIReItem to process
        config: Optional ensemble configuration
    
    Returns:
        EnsembleResult with aggregated prediction
    """
    results = {}
    for engine in engines:
        result = engine.rank_images(item)
        results[engine.name] = result
    
    aggregator = EnsembleAggregator(config)
    return aggregator.aggregate(results)


