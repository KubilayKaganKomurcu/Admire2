"""
AdMIRe 2.0 Evaluation Metrics
=============================
Implements evaluation metrics for the AdMIRe task.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AdMIReItem


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Ranking metrics
    dcg: float
    ndcg: float
    acc_at_1: float
    acc_at_3: float
    mrr: float  # Mean Reciprocal Rank
    
    # Sentence type metrics
    sentence_type_accuracy: float
    
    # Overall
    n_samples: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "DCG": self.dcg,
            "NDCG": self.ndcg,
            "Acc@1": self.acc_at_1,
            "Acc@3": self.acc_at_3,
            "MRR": self.mrr,
            "SentenceTypeAcc": self.sentence_type_accuracy,
            "N": self.n_samples
        }
    
    def __str__(self) -> str:
        return (
            f"Evaluation Results (n={self.n_samples}):\n"
            f"  DCG:      {self.dcg:.4f}\n"
            f"  NDCG:     {self.ndcg:.4f}\n"
            f"  Acc@1:    {self.acc_at_1:.4f}\n"
            f"  Acc@3:    {self.acc_at_3:.4f}\n"
            f"  MRR:      {self.mrr:.4f}\n"
            f"  Type Acc: {self.sentence_type_accuracy:.4f}"
        )


class AdMIReEvaluator:
    """
    Evaluator for AdMIRe 2.0 task.
    
    Computes standard ranking metrics:
    - DCG (Discounted Cumulative Gain)
    - NDCG (Normalized DCG)
    - Acc@k (Accuracy at position k)
    - MRR (Mean Reciprocal Rank)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.ground_truths = []
        self.sentence_type_preds = []
        self.sentence_type_gts = []
    
    def add_prediction(
        self,
        predicted_ranking: List[int],
        gold_ranking: List[str],
        image_names: List[str],
        predicted_type: Optional[str] = None,
        gold_type: Optional[str] = None
    ):
        """
        Add a single prediction for evaluation.
        
        Args:
            predicted_ranking: Predicted ranking [1,2,3,4,5] where pos i = rank of image i
            gold_ranking: Gold ordered list of image names (best first)
            image_names: List of image names in original order
            predicted_type: Predicted sentence type
            gold_type: Gold sentence type
        """
        # Convert gold ranking (image names) to numeric ranking
        gold_numeric = self._names_to_ranking(gold_ranking, image_names)
        
        self.predictions.append(predicted_ranking)
        self.ground_truths.append(gold_numeric)
        
        if predicted_type and gold_type:
            self.sentence_type_preds.append(predicted_type)
            self.sentence_type_gts.append(gold_type)
    
    def _names_to_ranking(
        self, 
        ordered_names: List[str], 
        all_names: List[str]
    ) -> List[int]:
        """
        Convert ordered list of names to ranking array.
        
        ordered_names: Names in order from best to worst
        all_names: Original list of all names
        
        Returns: ranking where ranking[i] = position of image i in gold order
        """
        ranking = [0] * len(all_names)
        
        for position, name in enumerate(ordered_names, 1):
            try:
                idx = all_names.index(name)
                ranking[idx] = position
            except ValueError:
                # Name not found, assign worst position
                continue
        
        # Fill any zeros with remaining positions
        used = set(ranking)
        remaining = [i for i in range(1, len(all_names) + 1) if i not in used]
        remaining_idx = 0
        for i in range(len(ranking)):
            if ranking[i] == 0:
                ranking[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        return ranking
    
    def compute_metrics(self) -> EvaluationResult:
        """Compute all metrics from accumulated predictions."""
        if not self.predictions:
            return EvaluationResult(
                dcg=0.0, ndcg=0.0, acc_at_1=0.0, acc_at_3=0.0,
                mrr=0.0, sentence_type_accuracy=0.0, n_samples=0
            )
        
        # Compute metrics
        dcg_scores = []
        ndcg_scores = []
        acc1_scores = []
        acc3_scores = []
        rr_scores = []  # Reciprocal ranks
        
        for pred, gold in zip(self.predictions, self.ground_truths):
            # Get predicted order (which image is at each position)
            pred_order = self._ranking_to_order(pred)
            gold_order = self._ranking_to_order(gold)
            
            # DCG and NDCG
            relevance = self._compute_relevance(pred_order, gold_order)
            dcg = self._dcg(relevance)
            ideal_relevance = sorted(relevance, reverse=True)
            idcg = self._dcg(ideal_relevance)
            ndcg = dcg / idcg if idcg > 0 else 0
            
            dcg_scores.append(dcg)
            ndcg_scores.append(ndcg)
            
            # Acc@1: Is the top predicted image the actual best?
            acc1_scores.append(1.0 if pred_order[0] == gold_order[0] else 0.0)
            
            # Acc@3: Is the gold best image in top 3 predictions?
            acc3_scores.append(1.0 if gold_order[0] in pred_order[:3] else 0.0)
            
            # MRR: Reciprocal rank of the gold best image
            try:
                rank_of_best = pred_order.index(gold_order[0]) + 1
                rr_scores.append(1.0 / rank_of_best)
            except ValueError:
                rr_scores.append(0.0)
        
        # Sentence type accuracy
        type_acc = 0.0
        if self.sentence_type_preds:
            correct = sum(1 for p, g in zip(self.sentence_type_preds, self.sentence_type_gts) if p == g)
            type_acc = correct / len(self.sentence_type_preds)
        
        return EvaluationResult(
            dcg=np.mean(dcg_scores),
            ndcg=np.mean(ndcg_scores),
            acc_at_1=np.mean(acc1_scores),
            acc_at_3=np.mean(acc3_scores),
            mrr=np.mean(rr_scores),
            sentence_type_accuracy=type_acc,
            n_samples=len(self.predictions)
        )
    
    def _ranking_to_order(self, ranking: List[int]) -> List[int]:
        """
        Convert ranking array to ordered list of indices.
        
        ranking[i] = rank of item i
        Returns: list of item indices in order (best first)
        """
        # Create (rank, index) pairs and sort by rank
        paired = [(rank, idx) for idx, rank in enumerate(ranking)]
        paired.sort(key=lambda x: x[0])
        return [idx for _, idx in paired]
    
    def _compute_relevance(
        self, 
        pred_order: List[int], 
        gold_order: List[int]
    ) -> List[float]:
        """
        Compute relevance scores for predicted order.
        
        Relevance of item at position i = how high it should be in gold order.
        """
        n = len(gold_order)
        relevance = []
        
        for pred_item in pred_order:
            try:
                gold_position = gold_order.index(pred_item)
                # Higher relevance for items that should be ranked higher
                rel = (n - gold_position) / n
            except ValueError:
                rel = 0.0
            relevance.append(rel)
        
        return relevance
    
    def _dcg(self, relevance: List[float], k: int = 5) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevance[:k]):
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        return dcg
    
    def evaluate_single(
        self,
        predicted_ranking: List[int],
        item: AdMIReItem,
        predicted_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction.
        
        Returns dict with individual metrics.
        """
        if not item.expected_order:
            return {"error": "No gold ranking available"}
        
        gold_ranking = self._names_to_ranking(item.expected_order, item.image_names)
        
        pred_order = self._ranking_to_order(predicted_ranking)
        gold_order = self._ranking_to_order(gold_ranking)
        
        relevance = self._compute_relevance(pred_order, gold_order)
        dcg = self._dcg(relevance)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = self._dcg(ideal_relevance)
        
        result = {
            "dcg": dcg,
            "ndcg": dcg / idcg if idcg > 0 else 0,
            "acc_at_1": 1.0 if pred_order[0] == gold_order[0] else 0.0,
            "acc_at_3": 1.0 if gold_order[0] in pred_order[:3] else 0.0,
        }
        
        if predicted_type and item.sentence_type:
            result["type_correct"] = predicted_type == item.sentence_type
        
        return result


def compare_rankings(
    ranking1: List[int],
    ranking2: List[int]
) -> Dict[str, float]:
    """
    Compare two rankings for agreement metrics.
    
    Returns:
        Dict with Kendall's tau, Spearman correlation, exact match positions
    """
    from scipy import stats
    
    # Convert rankings to orders for correlation
    order1 = [0] * len(ranking1)
    order2 = [0] * len(ranking2)
    
    for idx, rank in enumerate(ranking1):
        order1[rank - 1] = idx
    for idx, rank in enumerate(ranking2):
        order2[rank - 1] = idx
    
    # Kendall's tau
    tau, _ = stats.kendalltau(order1, order2)
    
    # Spearman correlation
    rho, _ = stats.spearmanr(ranking1, ranking2)
    
    # Exact position matches
    exact_matches = sum(1 for a, b in zip(ranking1, ranking2) if a == b)
    
    return {
        "kendall_tau": tau,
        "spearman_rho": rho,
        "exact_matches": exact_matches,
        "match_ratio": exact_matches / len(ranking1)
    }

