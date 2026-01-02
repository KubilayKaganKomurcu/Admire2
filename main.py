"""
AdMIRe 2.0 Main Entry Point
===========================
Unified system combining MIRA, DAALFT, and CTYUN-Lite engines
for multimodal idiomaticity ranking.

Usage:
    python main.py --mode demo           # Run with mock data
    python main.py --mode evaluate       # Evaluate on dataset
    python main.py --mode predict        # Generate predictions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AdMIRe2Config, get_config
from data_loader import AdMIReDataLoader, create_mock_data, AdMIReItem
from engines import MIRAEngine, DAALFTEngine, CTYUNLiteEngine
from evaluation.ensemble import EnsembleAggregator, EnsembleResult
from evaluation.metrics import AdMIReEvaluator


class AdMIRe2System:
    """
    Main system class that orchestrates the ensemble.
    
    Combines three engines:
    - MIRA: Self-consistency with multi-step reasoning
    - DAALFT: Detect → Explain → Rank pipeline
    - CTYUN-Lite: Caption-based direct ranking
    """
    
    def __init__(self, config: Optional[AdMIRe2Config] = None):
        self.config = config or get_config()
        
        # Initialize engines based on config
        self.engines = {}
        if "mira" in self.config.ensemble.enabled_engines:
            self.engines["MIRA"] = MIRAEngine(self.config)
        if "daalft" in self.config.ensemble.enabled_engines:
            self.engines["DAALFT"] = DAALFTEngine(self.config)
        if "ctyun_lite" in self.config.ensemble.enabled_engines:
            self.engines["CTYUN-Lite"] = CTYUNLiteEngine(self.config)
        
        # Initialize aggregator and evaluator
        self.aggregator = EnsembleAggregator(self.config.ensemble)
        self.evaluator = AdMIReEvaluator()
        
        # Data loader
        self.data_loader = AdMIReDataLoader(self.config.data.base_path)
        
        if self.config.verbose:
            print(f"AdMIRe 2.0 System initialized with engines: {list(self.engines.keys())}")
    
    def predict_single(
        self, 
        item: AdMIReItem,
        use_ensemble: bool = True
    ) -> EnsembleResult:
        """
        Generate prediction for a single item.
        
        Args:
            item: AdMIReItem to process
            use_ensemble: If True, use ensemble; else use first engine only
        
        Returns:
            EnsembleResult with ranking and sentence type
        """
        results = {}
        
        for name, engine in self.engines.items():
            if self.config.verbose:
                print(f"  Running {name}...")
            result = engine.rank_images(item)
            results[name] = result
            
            if not use_ensemble:
                # Return first engine's result wrapped as ensemble
                return EnsembleResult(
                    ranking=result.ranking,
                    sentence_type=result.sentence_type,
                    ranking_confidence=result.ranking_confidence,
                    sentence_type_confidence=result.sentence_type_confidence,
                    engine_results=results,
                    aggregation_method="single",
                    agreement_score=1.0
                )
        
        # Aggregate results
        ensemble_result = self.aggregator.aggregate(results)
        return ensemble_result
    
    def predict_batch(
        self,
        items: List[AdMIReItem],
        use_ensemble: bool = True,
        show_progress: bool = True
    ) -> List[EnsembleResult]:
        """
        Generate predictions for multiple items.
        """
        results = []
        iterator = tqdm(items, desc="Processing") if show_progress else items
        
        for item in iterator:
            result = self.predict_single(item, use_ensemble)
            results.append(result)
        
        return results
    
    def evaluate(
        self,
        language: str = "english",
        split: str = "dev",
        use_ensemble: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on a dataset split.
        
        Args:
            language: Language to evaluate ("english" or "portuguese")
            split: Data split ("train", "dev", "test")
            use_ensemble: Whether to use ensemble or single engine
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Load data
        items = self.data_loader.load_subtask_a(language, split)
        
        if not items:
            print(f"No data found for {language}/{split}")
            return {}
        
        print(f"Evaluating on {len(items)} items from {language}/{split}")
        
        # Reset evaluator
        self.evaluator.reset()
        
        # Process each item
        for item in tqdm(items, desc="Evaluating"):
            result = self.predict_single(item, use_ensemble)
            
            if item.expected_order:
                self.evaluator.add_prediction(
                    predicted_ranking=result.ranking,
                    gold_ranking=item.expected_order,
                    image_names=item.image_names,
                    predicted_type=result.sentence_type,
                    gold_type=item.sentence_type
                )
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics()
        print(metrics)
        
        return metrics.to_dict()
    
    def generate_submission(
        self,
        language: str = "english",
        split: str = "test",
        output_file: str = "submission.tsv"
    ):
        """
        Generate submission file for the shared task.
        """
        items = self.data_loader.load_subtask_a(language, split)
        
        if not items:
            print(f"No data found for {language}/{split}")
            return
        
        print(f"Generating predictions for {len(items)} items")
        
        results = []
        for item in tqdm(items, desc="Predicting"):
            ensemble = self.predict_single(item, use_ensemble=True)
            
            # Convert ranking to ordered image names
            ordered_images = ensemble.to_ordered_images(item.image_names)
            
            results.append({
                "compound": item.compound,
                "predicted_order": ",".join(ordered_images),
                "sentence_type": ensemble.sentence_type
            })
        
        # Save to file
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)
        print(f"Submission saved to {output_file}")


def run_demo():
    """Run demo with mock data."""
    print("=" * 60)
    print("AdMIRe 2.0 Demo - Running with Mock Data")
    print("=" * 60)
    
    # Create system
    system = AdMIRe2System()
    
    # Load mock data
    mock_items = create_mock_data()
    
    print(f"\nProcessing {len(mock_items)} mock items...\n")
    
    for i, item in enumerate(mock_items, 1):
        print(f"\n{'='*50}")
        print(f"Item {i}: '{item.compound}'")
        print(f"Sentence: {item.sentence}")
        print(f"Gold type: {item.sentence_type}")
        print(f"{'='*50}")
        
        # Get prediction
        result = system.predict_single(item)
        
        print(f"\n📊 Ensemble Result:")
        print(f"  Sentence type: {result.sentence_type} (conf: {result.sentence_type_confidence:.2f})")
        print(f"  Ranking: {result.ranking}")
        print(f"  Agreement score: {result.agreement_score:.2f}")
        
        # Show per-engine results
        print(f"\n📋 Per-Engine Results:")
        for engine_name, engine_result in result.engine_results.items():
            print(f"  {engine_name}:")
            print(f"    - Type: {engine_result.sentence_type}")
            print(f"    - Ranking: {engine_result.ranking}")
            print(f"    - Time: {engine_result.processing_time:.2f}s")
        
        # Compare with gold
        if item.expected_order:
            ordered_pred = result.to_ordered_images(item.image_names)
            print(f"\n🎯 Comparison:")
            print(f"  Gold order:      {item.expected_order}")
            print(f"  Predicted order: {ordered_pred}")
            
            # Quick accuracy check
            matches = sum(1 for a, b in zip(item.expected_order, ordered_pred) if a == b)
            print(f"  Exact matches: {matches}/5")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="AdMIRe 2.0 Ensemble System")
    parser.add_argument(
        "--mode", 
        choices=["demo", "evaluate", "predict"],
        default="demo",
        help="Operation mode"
    )
    parser.add_argument("--language", default="english", help="Language to process")
    parser.add_argument("--split", default="dev", help="Data split")
    parser.add_argument("--output", default="submission.tsv", help="Output file for predictions")
    parser.add_argument("--single-engine", type=str, help="Use single engine instead of ensemble")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    
    elif args.mode == "evaluate":
        system = AdMIRe2System()
        system.evaluate(args.language, args.split)
    
    elif args.mode == "predict":
        system = AdMIRe2System()
        system.generate_submission(args.language, args.split, args.output)


if __name__ == "__main__":
    main()





