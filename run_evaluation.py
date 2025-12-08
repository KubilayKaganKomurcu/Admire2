#!/usr/bin/env python3
"""
AdMIRe 2.0 Evaluation Script
============================
Runs all implemented methods on the Turkish dataset and logs detailed metrics.

Metrics computed:
- Top-1 Accuracy (Acc@1): Is the best predicted image correct?
- First Place Exact Match: Does predicted rank 1 match gold rank 1?
- Second Place Exact Match: Does predicted rank 2 match gold rank 2?
- NDCG, MRR, DCG for overall ranking quality

Usage:
    python run_evaluation.py --language Turkish --limit 10
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AdMIRe2Config, get_config
from data_loader import AdMIReItem
from engines import MIRAEngine, DAALFTEngine, CTYUNLiteEngine
from engines.base_engine import EngineResult
from evaluation.ensemble import EnsembleAggregator
from evaluation.metrics import AdMIReEvaluator


@dataclass
class DetailedMetrics:
    """Detailed metrics for a single method."""
    name: str
    n_samples: int = 0
    
    # Position-specific exact matches
    first_place_exact_match: float = 0.0   # Predicted rank 1 == Gold rank 1
    second_place_exact_match: float = 0.0  # Predicted rank 2 == Gold rank 2
    third_place_exact_match: float = 0.0   # Predicted rank 3 == Gold rank 3
    
    # Standard metrics
    acc_at_1: float = 0.0  # Top-1 accuracy (same as first place exact match conceptually)
    acc_at_3: float = 0.0  # Gold best in top 3
    mrr: float = 0.0       # Mean Reciprocal Rank
    ndcg: float = 0.0      # Normalized DCG
    dcg: float = 0.0       # Discounted Cumulative Gain
    
    # Timing
    avg_time_per_sample: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "first_place_exact_match": self.first_place_exact_match,
            "second_place_exact_match": self.second_place_exact_match,
            "third_place_exact_match": self.third_place_exact_match,
            "acc_at_1": self.acc_at_1,
            "acc_at_3": self.acc_at_3,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "dcg": self.dcg,
            "avg_time_per_sample": self.avg_time_per_sample,
            "total_time": self.total_time
        }
    
    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"📊 {self.name} Results (n={self.n_samples})\n"
            f"{'='*60}\n"
            f"  Position Exact Matches:\n"
            f"    1st Place: {self.first_place_exact_match:.4f} ({self.first_place_exact_match*100:.1f}%)\n"
            f"    2nd Place: {self.second_place_exact_match:.4f} ({self.second_place_exact_match*100:.1f}%)\n"
            f"    3rd Place: {self.third_place_exact_match:.4f} ({self.third_place_exact_match*100:.1f}%)\n"
            f"\n  Ranking Metrics:\n"
            f"    Acc@1:    {self.acc_at_1:.4f} ({self.acc_at_1*100:.1f}%)\n"
            f"    Acc@3:    {self.acc_at_3:.4f} ({self.acc_at_3*100:.1f}%)\n"
            f"    MRR:      {self.mrr:.4f}\n"
            f"    NDCG:     {self.ndcg:.4f}\n"
            f"    DCG:      {self.dcg:.4f}\n"
            f"\n  Timing:\n"
            f"    Avg/sample: {self.avg_time_per_sample:.2f}s\n"
            f"    Total:      {self.total_time:.2f}s\n"
        )


class TurkishDataLoader:
    """
    Custom data loader for Turkish submission format.
    
    Loads from:
    - TSV: data/admire_file/submission_Turkish.tsv
    - Images: data/admire2_data/Turkish/{compound}/{image}.png
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.tsv_path = self.base_path / "admire_file" / "submission_Turkish.tsv"
        self.images_path = self.base_path / "admire2_data" / "Turkish"
    
    def load(self, limit: Optional[int] = None) -> List[AdMIReItem]:
        """
        Load Turkish dataset items.
        
        Args:
            limit: Maximum number of items to load (None for all)
        
        Returns:
            List of AdMIReItem objects
        """
        if not self.tsv_path.exists():
            print(f"❌ TSV file not found: {self.tsv_path}")
            return []
        
        df = pd.read_csv(self.tsv_path, sep='\t')
        
        if limit:
            df = df.head(limit)
        
        items = []
        skipped = 0
        
        for idx, row in df.iterrows():
            compound = row["compound"]
            sentence = row["sentence"]
            
            # Extract image names and captions
            image_names = [row[f"image{i}_name"] for i in range(1, 6)]
            image_captions = [str(row[f"image{i}_caption"]) for i in range(1, 6)]
            
            # Build image paths
            compound_folder = self.images_path / compound
            image_paths = [str(compound_folder / name) for name in image_names]
            
            # Check if images exist
            images_exist = all(Path(p).exists() for p in image_paths)
            
            # Parse expected order if available
            expected_order = None
            if "expected_order" in df.columns and pd.notna(row.get("expected_order")):
                order_str = str(row["expected_order"]).strip()
                if order_str:
                    order_str = order_str.strip("[]")
                    expected_order = [s.strip().strip("'\"") for s in order_str.split(",")]
            
            # If no gold labels, skip for evaluation (or use for prediction only)
            # For now, we'll include items even without gold labels
            
            item = AdMIReItem(
                compound=compound,
                sentence=sentence,
                image_names=image_names,
                image_captions=image_captions,
                image_paths=image_paths,
                sentence_type=row.get("sentence_type"),
                expected_order=expected_order,
                subset="test",
                language="tr",
                item_id=f"turkish_{idx}"
            )
            
            items.append(item)
        
        print(f"✅ Loaded {len(items)} items from Turkish dataset")
        if not self.images_path.exists():
            print(f"⚠️  Images folder not found: {self.images_path}")
            print("   Evaluation will run in caption-only mode.")
        
        return items


class MethodEvaluator:
    """
    Evaluates individual methods and ensemble on the dataset.
    """
    
    def __init__(self, config: Optional[AdMIRe2Config] = None):
        self.config = config or get_config()
        
        # Initialize engines
        self.engines = {
            "MIRA": MIRAEngine(self.config),
            "DAALFT": DAALFTEngine(self.config),
            "CTYUN-Lite": CTYUNLiteEngine(self.config)
        }
        
        # Ensemble aggregator
        self.aggregator = EnsembleAggregator(self.config.ensemble)
    
    def evaluate_method(
        self, 
        method_name: str, 
        items: List[AdMIReItem],
        show_progress: bool = True
    ) -> Tuple[DetailedMetrics, List[EngineResult]]:
        """
        Evaluate a single method on the dataset.
        
        Returns:
            Tuple of (DetailedMetrics, list of EngineResult)
        """
        import time
        
        if method_name not in self.engines:
            raise ValueError(f"Unknown method: {method_name}")
        
        engine = self.engines[method_name]
        metrics = DetailedMetrics(name=method_name)
        results = []
        
        # Counters for metrics
        first_matches = 0
        second_matches = 0
        third_matches = 0
        acc1_sum = 0.0
        acc3_sum = 0.0
        rr_sum = 0.0
        ndcg_sum = 0.0
        dcg_sum = 0.0
        evaluated = 0
        
        iterator = tqdm(items, desc=f"Running {method_name}") if show_progress else items
        start_time = time.time()
        
        for item in iterator:
            try:
                result = engine.rank_images(item)
                results.append(result)
                
                # Skip evaluation if no gold labels
                if not item.expected_order:
                    continue
                
                # Convert to comparable formats
                pred_order = self._ranking_to_ordered_names(result.ranking, item.image_names)
                gold_order = item.expected_order
                
                # Position exact matches
                if len(pred_order) >= 1 and len(gold_order) >= 1:
                    if pred_order[0] == gold_order[0]:
                        first_matches += 1
                        acc1_sum += 1.0
                
                if len(pred_order) >= 2 and len(gold_order) >= 2:
                    if pred_order[1] == gold_order[1]:
                        second_matches += 1
                
                if len(pred_order) >= 3 and len(gold_order) >= 3:
                    if pred_order[2] == gold_order[2]:
                        third_matches += 1
                
                # Acc@3: Is gold best in top 3?
                if gold_order[0] in pred_order[:3]:
                    acc3_sum += 1.0
                
                # MRR: Reciprocal rank of gold best
                try:
                    rank_of_best = pred_order.index(gold_order[0]) + 1
                    rr_sum += 1.0 / rank_of_best
                except ValueError:
                    pass
                
                # NDCG computation
                relevance = self._compute_relevance(pred_order, gold_order)
                dcg = self._dcg(relevance)
                ideal = sorted(relevance, reverse=True)
                idcg = self._dcg(ideal)
                ndcg = dcg / idcg if idcg > 0 else 0
                
                dcg_sum += dcg
                ndcg_sum += ndcg
                evaluated += 1
                
            except Exception as e:
                print(f"Error processing item {item.item_id}: {e}")
                results.append(None)
        
        total_time = time.time() - start_time
        
        # Compute final metrics
        if evaluated > 0:
            metrics.n_samples = evaluated
            metrics.first_place_exact_match = first_matches / evaluated
            metrics.second_place_exact_match = second_matches / evaluated
            metrics.third_place_exact_match = third_matches / evaluated
            metrics.acc_at_1 = acc1_sum / evaluated
            metrics.acc_at_3 = acc3_sum / evaluated
            metrics.mrr = rr_sum / evaluated
            metrics.ndcg = ndcg_sum / evaluated
            metrics.dcg = dcg_sum / evaluated
        
        metrics.total_time = total_time
        metrics.avg_time_per_sample = total_time / len(items) if items else 0
        
        return metrics, results
    
    def evaluate_ensemble(
        self,
        items: List[AdMIReItem],
        method_results: Dict[str, List[EngineResult]],
        show_progress: bool = True
    ) -> DetailedMetrics:
        """
        Evaluate ensemble of all methods.
        """
        import time
        
        metrics = DetailedMetrics(name="ENSEMBLE")
        
        first_matches = 0
        second_matches = 0
        third_matches = 0
        acc1_sum = 0.0
        acc3_sum = 0.0
        rr_sum = 0.0
        ndcg_sum = 0.0
        dcg_sum = 0.0
        evaluated = 0
        
        start_time = time.time()
        iterator = tqdm(range(len(items)), desc="Computing Ensemble") if show_progress else range(len(items))
        
        for i in iterator:
            item = items[i]
            
            # Collect results from all methods for this item
            engine_results = {}
            for method_name, results in method_results.items():
                if results[i] is not None:
                    engine_results[method_name] = results[i]
            
            if not engine_results:
                continue
            
            # Aggregate
            ensemble_result = self.aggregator.aggregate(engine_results)
            
            # Skip if no gold labels
            if not item.expected_order:
                continue
            
            # Convert to comparable formats
            pred_order = self._ranking_to_ordered_names(ensemble_result.ranking, item.image_names)
            gold_order = item.expected_order
            
            # Position exact matches
            if len(pred_order) >= 1 and len(gold_order) >= 1:
                if pred_order[0] == gold_order[0]:
                    first_matches += 1
                    acc1_sum += 1.0
            
            if len(pred_order) >= 2 and len(gold_order) >= 2:
                if pred_order[1] == gold_order[1]:
                    second_matches += 1
            
            if len(pred_order) >= 3 and len(gold_order) >= 3:
                if pred_order[2] == gold_order[2]:
                    third_matches += 1
            
            # Acc@3
            if gold_order[0] in pred_order[:3]:
                acc3_sum += 1.0
            
            # MRR
            try:
                rank_of_best = pred_order.index(gold_order[0]) + 1
                rr_sum += 1.0 / rank_of_best
            except ValueError:
                pass
            
            # NDCG
            relevance = self._compute_relevance(pred_order, gold_order)
            dcg = self._dcg(relevance)
            ideal = sorted(relevance, reverse=True)
            idcg = self._dcg(ideal)
            ndcg = dcg / idcg if idcg > 0 else 0
            
            dcg_sum += dcg
            ndcg_sum += ndcg
            evaluated += 1
        
        total_time = time.time() - start_time
        
        if evaluated > 0:
            metrics.n_samples = evaluated
            metrics.first_place_exact_match = first_matches / evaluated
            metrics.second_place_exact_match = second_matches / evaluated
            metrics.third_place_exact_match = third_matches / evaluated
            metrics.acc_at_1 = acc1_sum / evaluated
            metrics.acc_at_3 = acc3_sum / evaluated
            metrics.mrr = rr_sum / evaluated
            metrics.ndcg = ndcg_sum / evaluated
            metrics.dcg = dcg_sum / evaluated
        
        metrics.total_time = total_time
        metrics.avg_time_per_sample = total_time / len(items) if items else 0
        
        return metrics
    
    def _ranking_to_ordered_names(self, ranking: List[int], names: List[str]) -> List[str]:
        """Convert ranking array to ordered list of names."""
        paired = list(zip(ranking, names))
        paired.sort(key=lambda x: x[0])
        return [name for _, name in paired]
    
    def _compute_relevance(self, pred_order: List[str], gold_order: List[str]) -> List[float]:
        """Compute relevance scores."""
        n = len(gold_order)
        relevance = []
        for item in pred_order:
            try:
                gold_pos = gold_order.index(item)
                rel = (n - gold_pos) / n
            except ValueError:
                rel = 0.0
            relevance.append(rel)
        return relevance
    
    def _dcg(self, relevance: List[float], k: int = 5) -> float:
        """Compute DCG."""
        import math
        dcg = 0.0
        for i, rel in enumerate(relevance[:k]):
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        return dcg


def run_full_evaluation(
    language: str = "Turkish",
    limit: Optional[int] = None,
    output_file: Optional[str] = None
):
    """
    Run full evaluation on all methods and ensemble.
    
    Args:
        language: Language to evaluate (currently only Turkish supported)
        limit: Limit number of samples (for testing)
        output_file: Path to save results JSON
    """
    print("\n" + "="*70)
    print("🚀 AdMIRe 2.0 Full Evaluation")
    print("="*70)
    print(f"Language: {language}")
    print(f"Limit: {limit if limit else 'None (all samples)'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Load data
    loader = TurkishDataLoader()
    items = loader.load(limit=limit)
    
    if not items:
        print("❌ No data loaded. Exiting.")
        return
    
    # Check for gold labels
    items_with_gold = [item for item in items if item.expected_order]
    print(f"\n📋 Items with gold labels: {len(items_with_gold)}/{len(items)}")
    
    if not items_with_gold:
        print("\n⚠️  No gold labels available. Running prediction-only mode...")
        print("   To compute metrics, ensure 'expected_order' column has values.")
    
    # Initialize evaluator
    evaluator = MethodEvaluator()
    
    # Store all results
    all_metrics = {}
    all_results = {}
    
    # Evaluate each method
    methods = ["MIRA", "DAALFT", "CTYUN-Lite"]
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"🔧 Evaluating {method}...")
        print("="*60)
        
        metrics, results = evaluator.evaluate_method(method, items)
        all_metrics[method] = metrics
        all_results[method] = results
        
        print(metrics)
    
    # Evaluate ensemble
    print(f"\n{'='*60}")
    print("🔗 Computing Ensemble...")
    print("="*60)
    
    ensemble_metrics = evaluator.evaluate_ensemble(items, all_results)
    all_metrics["ENSEMBLE"] = ensemble_metrics
    
    print(ensemble_metrics)
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Method':<15} {'1st Match':<12} {'2nd Match':<12} {'Acc@1':<10} {'MRR':<10} {'NDCG':<10}")
    print("-"*70)
    
    for name, metrics in all_metrics.items():
        print(f"{name:<15} {metrics.first_place_exact_match:.4f}       "
              f"{metrics.second_place_exact_match:.4f}       "
              f"{metrics.acc_at_1:.4f}     "
              f"{metrics.mrr:.4f}     "
              f"{metrics.ndcg:.4f}")
    
    print("="*70)
    
    # Save results
    if output_file:
        results_dict = {
            "metadata": {
                "language": language,
                "n_samples": len(items),
                "n_with_gold": len(items_with_gold),
                "timestamp": datetime.now().isoformat()
            },
            "metrics": {name: m.to_dict() for name, m in all_metrics.items()}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="AdMIRe 2.0 Evaluation Script")
    parser.add_argument("--language", default="Turkish", help="Language to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    run_full_evaluation(
        language=args.language,
        limit=args.limit,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

