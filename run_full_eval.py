"""
AdMIRe 2.0 Full Evaluation Script
=================================
Comprehensive evaluation on EN and PT data with full logging.

Usage:
    python run_full_eval.py --language EN --max_samples 10
    python run_full_eval.py --language PT --subset Dev
    python run_full_eval.py --language EN --all
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import asdict

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AdMIRe2Config, get_config
from data_loader import AdMIReItem
from engines import MIRAEngine, DAALFTEngine, CTYUNLiteEngine
from evaluation.ensemble import EnsembleAggregator, EnsembleResult
from evaluation.metrics import AdMIReEvaluator


class FullLogger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.file = open(log_file, 'w', encoding='utf-8')
        self.log(f"=" * 80)
        self.log(f"AdMIRe 2.0 Full Evaluation Log")
        self.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"=" * 80)
    
    def log(self, message: str = ""):
        print(message)
        self.file.write(message + "\n")
        self.file.flush()
    
    def close(self):
        self.log(f"\n{'=' * 80}")
        self.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Log saved to: {self.log_file}")
        self.file.close()


def load_language_data(
    language: str,
    base_path: str = "data",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None
) -> List[AdMIReItem]:
    """
    Load language data with gold labels.
    
    Args:
        language: "EN", "PT", "Turkish", etc.
        base_path: Path to data folder
        subset: Filter by subset ("Dev", "Train", "Test") or None for all
        max_samples: Limit number of samples
    """
    base = Path(base_path)
    
    # Handle different TSV naming conventions
    if language in ["EN", "PT"]:
        tsv_file = base / "TSVs" / f"{language}_subtask_a.tsv"
    else:
        tsv_file = base / "TSVs" / f"submission_{language}.tsv"
    
    images_base = base / "languages" / language
    
    if not tsv_file.exists():
        print(f"Error: TSV file not found at {tsv_file}")
        return []
    
    # Read TSV
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"Found {len(df)} items in {language} dataset")
    print(f"Columns: {list(df.columns)}")
    
    # Filter by subset if specified
    if subset and "subset" in df.columns:
        df = df[df["subset"].str.lower() == subset.lower()]
        print(f"Filtered to {len(df)} items with subset={subset}")
    
    # Limit samples if requested
    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to first {max_samples} samples")
    
    items = []
    for idx, row in df.iterrows():
        compound = row["compound"]
        
        # Extract image names and captions
        image_names = []
        image_captions = []
        for i in range(1, 6):
            name_col = f"image{i}_name"
            caption_col = f"image{i}_caption"
            
            if name_col in df.columns and pd.notna(row.get(name_col)):
                image_names.append(str(row[name_col]))
            else:
                image_names.append(f"image{i}.png")
            
            if caption_col in df.columns and pd.notna(row.get(caption_col)):
                image_captions.append(str(row[caption_col]))
            else:
                image_captions.append("")
        
        # Build image paths
        compound_folder = images_base / compound
        image_paths = [str(compound_folder / name) for name in image_names]
        
        # Parse expected order (gold ranking)
        expected_order = None
        if "expected_order" in df.columns and pd.notna(row.get("expected_order")):
            order_str = str(row["expected_order"])
            order_str = order_str.strip("[]")
            expected_order = [s.strip().strip("'\"") for s in order_str.split(",")]
        
        # Get sentence type
        sentence_type = None
        if "sentence_type" in df.columns and pd.notna(row.get("sentence_type")):
            sentence_type = str(row["sentence_type"]).lower()
        
        item = AdMIReItem(
            compound=compound,
            sentence=str(row.get("sentence", "")),
            image_names=image_names,
            image_captions=image_captions,
            image_paths=image_paths,
            sentence_type=sentence_type,
            expected_order=expected_order,
            subset=str(row.get("subset", "unknown")),
            language=language.lower(),
            item_id=f"{language}_{idx}"
        )
        items.append(item)
    
    return items


def run_full_evaluation(
    language: str = "Turkish",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "eval_results",
    text_only: bool = False
):
    """Run comprehensive evaluation with full logging."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_textonly" if text_only else ""
    log_file = os.path.join(output_dir, f"eval_{language}{mode_suffix}_{timestamp}.log")
    results_file = os.path.join(output_dir, f"eval_{language}{mode_suffix}_{timestamp}.json")
    
    logger = FullLogger(log_file)
    config = get_config()
    
    # Log configuration
    logger.log(f"\n{'=' * 80}")
    logger.log("CONFIGURATION")
    logger.log(f"{'=' * 80}")
    logger.log(f"Language: {language}")
    logger.log(f"Subset filter: {subset or 'All'}")
    logger.log(f"Max samples: {max_samples or 'All'}")
    logger.log(f"Mode: {'TEXT-ONLY (captions)' if text_only else 'MULTIMODAL (images + captions)'}")
    logger.log(f"Vision model: {config.api.vision_model}")
    logger.log(f"Text model: {config.api.text_model}")
    logger.log(f"Enabled engines: {config.ensemble.enabled_engines}")
    logger.log(f"Aggregation method: {config.ensemble.aggregation_method}")
    
    # Load data
    logger.log(f"\n{'=' * 80}")
    logger.log("LOADING DATA")
    logger.log(f"{'=' * 80}")
    
    items = load_language_data(language, subset=subset, max_samples=max_samples)
    
    if not items:
        logger.log("ERROR: No data loaded!")
        logger.close()
        return
    
    logger.log(f"Loaded {len(items)} items")
    
    # Count items with gold labels
    items_with_gold = sum(1 for item in items if item.expected_order)
    items_with_type = sum(1 for item in items if item.sentence_type)
    logger.log(f"Items with gold ranking: {items_with_gold}")
    logger.log(f"Items with gold sentence type: {items_with_type}")
    
    # Initialize engines
    logger.log(f"\n{'=' * 80}")
    logger.log("INITIALIZING ENGINES")
    logger.log(f"{'=' * 80}")
    
    engines = {}
    if "mira" in config.ensemble.enabled_engines:
        engines["MIRA"] = MIRAEngine(config)
        logger.log(f"✓ MIRA engine loaded (samples={config.engine.mira_num_samples})")
    if "daalft" in config.ensemble.enabled_engines:
        engines["DAALFT"] = DAALFTEngine(config)
        logger.log(f"✓ DAALFT engine loaded")
    if "ctyun_lite" in config.ensemble.enabled_engines:
        engines["CTYUN-Lite"] = CTYUNLiteEngine(config)
        logger.log(f"✓ CTYUN-Lite engine loaded")
    
    aggregator = EnsembleAggregator(config.ensemble)
    evaluator = AdMIReEvaluator()
    
    # Store all results
    all_results = []
    
    # Process each item
    logger.log(f"\n{'=' * 80}")
    logger.log("EVALUATION")
    logger.log(f"{'=' * 80}")
    
    for i, item in enumerate(items, 1):
        logger.log(f"\n{'─' * 80}")
        logger.log(f"ITEM {i}/{len(items)}: '{item.compound}'")
        logger.log(f"{'─' * 80}")
        logger.log(f"Sentence: {item.sentence}")
        logger.log(f"Subset: {item.subset}")
        logger.log(f"Gold sentence type: {item.sentence_type or 'N/A'}")
        logger.log(f"Gold order: {item.expected_order or 'N/A'}")
        logger.log(f"Image names: {item.image_names}")
        
        # Force text-only mode by clearing image paths
        if text_only:
            item.image_paths = []  # This forces caption-only mode in engines
        
        # Log captions
        logger.log(f"\nCaptions:")
        for j, cap in enumerate(item.image_captions, 1):
            cap_preview = cap[:80] + "..." if len(cap) > 80 else cap
            logger.log(f"  [{j}] {cap_preview}")
        
        # Run each engine
        item_result = {
            "item_id": item.item_id,
            "compound": item.compound,
            "sentence": item.sentence,
            "subset": item.subset,
            "gold_type": item.sentence_type,
            "gold_order": item.expected_order,
            "image_names": item.image_names,
            "engine_results": {},
            "ensemble_result": None
        }
        
        engine_results = {}
        
        for name, engine in engines.items():
            logger.log(f"\n  → Running {name}...")
            try:
                result = engine.rank_images(item)
                engine_results[name] = result
                
                # Log detailed results
                logger.log(f"    Sentence type: {result.sentence_type} (conf: {result.sentence_type_confidence:.2f})")
                logger.log(f"    Ranking: {result.ranking}")
                if result.image_scores:
                    logger.log(f"    Image scores: {[f'{s:.2f}' for s in result.image_scores]}")
                logger.log(f"    Processing time: {result.processing_time:.2f}s")
                
                # Store for JSON
                item_result["engine_results"][name] = {
                    "sentence_type": result.sentence_type,
                    "sentence_type_confidence": result.sentence_type_confidence,
                    "ranking": result.ranking,
                    "ranking_confidence": result.ranking_confidence,
                    "image_scores": result.image_scores,
                    "processing_time": result.processing_time,
                    "reasoning": result.reasoning[:500] if result.reasoning else None
                }
                
            except Exception as e:
                logger.log(f"    ERROR: {e}")
                item_result["engine_results"][name] = {"error": str(e)}
        
        # Ensemble aggregation
        if engine_results:
            logger.log(f"\n  → Ensemble Aggregation ({config.ensemble.aggregation_method}):")
            
            ensemble_result = aggregator.aggregate(engine_results)
            
            logger.log(f"    Final sentence type: {ensemble_result.sentence_type} (conf: {ensemble_result.sentence_type_confidence:.2f})")
            logger.log(f"    Final ranking: {ensemble_result.ranking}")
            logger.log(f"    Agreement score: {ensemble_result.agreement_score:.2f}")
            
            # Convert ranking to image order
            ordered_images = ensemble_result.to_ordered_images(item.image_names)
            logger.log(f"    Predicted order: {ordered_images}")
            
            item_result["ensemble_result"] = {
                "sentence_type": ensemble_result.sentence_type,
                "sentence_type_confidence": ensemble_result.sentence_type_confidence,
                "ranking": ensemble_result.ranking,
                "ranking_confidence": ensemble_result.ranking_confidence,
                "agreement_score": ensemble_result.agreement_score,
                "predicted_order": ordered_images
            }
            
            # Compare with gold
            if item.expected_order:
                logger.log(f"\n  → Comparison with Gold:")
                logger.log(f"    Gold order:      {item.expected_order}")
                logger.log(f"    Predicted order: {ordered_images}")
                
                matches = sum(1 for a, b in zip(item.expected_order, ordered_images) if a == b)
                logger.log(f"    Exact position matches: {matches}/5")
                
                # Check if top-1 is correct
                top1_correct = ordered_images[0] == item.expected_order[0] if item.expected_order else False
                logger.log(f"    Top-1 correct: {top1_correct}")
                
                item_result["comparison"] = {
                    "exact_matches": matches,
                    "top1_correct": top1_correct
                }
                
                # Add to evaluator
                evaluator.add_prediction(
                    predicted_ranking=ensemble_result.ranking,
                    gold_ranking=item.expected_order,
                    image_names=item.image_names,
                    predicted_type=ensemble_result.sentence_type,
                    gold_type=item.sentence_type
                )
            
            # Check sentence type
            if item.sentence_type:
                type_correct = ensemble_result.sentence_type == item.sentence_type
                logger.log(f"    Sentence type correct: {type_correct} (pred={ensemble_result.sentence_type}, gold={item.sentence_type})")
                item_result["type_correct"] = type_correct
        
        all_results.append(item_result)
    
    # Compute final metrics
    logger.log(f"\n{'=' * 80}")
    logger.log("FINAL METRICS")
    logger.log(f"{'=' * 80}")
    
    metrics = evaluator.compute_metrics()
    logger.log(str(metrics))
    
    # Additional statistics
    logger.log(f"\nAdditional Statistics:")
    
    type_correct = sum(1 for r in all_results if r.get("type_correct", False))
    type_total = sum(1 for r in all_results if "type_correct" in r)
    if type_total > 0:
        logger.log(f"  Sentence type accuracy: {type_correct}/{type_total} = {100*type_correct/type_total:.1f}%")
    
    top1_correct = sum(1 for r in all_results if r.get("comparison", {}).get("top1_correct", False))
    top1_total = sum(1 for r in all_results if "comparison" in r)
    if top1_total > 0:
        logger.log(f"  Top-1 accuracy: {top1_correct}/{top1_total} = {100*top1_correct/top1_total:.1f}%")
    
    # Per-engine agreement statistics
    logger.log(f"\nPer-Engine Statistics:")
    for engine_name in engines.keys():
        engine_type_correct = 0
        engine_total = 0
        for r in all_results:
            if engine_name in r["engine_results"] and "sentence_type" in r["engine_results"][engine_name]:
                engine_total += 1
                if r.get("gold_type") and r["engine_results"][engine_name]["sentence_type"] == r["gold_type"]:
                    engine_type_correct += 1
        if engine_total > 0:
            logger.log(f"  {engine_name} sentence type accuracy: {engine_type_correct}/{engine_total} = {100*engine_type_correct/engine_total:.1f}%")
    
    # Save JSON results
    logger.log(f"\n{'=' * 80}")
    logger.log("SAVING RESULTS")
    logger.log(f"{'=' * 80}")
    
    output_data = {
        "metadata": {
            "language": language,
            "subset": subset,
            "max_samples": max_samples,
            "timestamp": timestamp,
            "config": {
                "vision_model": config.api.vision_model,
                "text_model": config.api.text_model,
                "enabled_engines": config.ensemble.enabled_engines,
                "aggregation_method": config.ensemble.aggregation_method
            }
        },
        "metrics": metrics.to_dict(),
        "results": all_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.log(f"Results saved to: {results_file}")
    logger.close()
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Full AdMIRe 2.0 Evaluation")
    parser.add_argument("--language", default="Turkish", 
                        help="Language to evaluate (EN, PT, Turkish, Chinese, etc.)")
    parser.add_argument("--subset", default=None, choices=["Dev", "Train", "Test"], help="Filter by subset")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory for logs and results")
    parser.add_argument("--all", action="store_true", help="Evaluate all samples (ignore max_samples)")
    parser.add_argument("--text-only", action="store_true", dest="text_only",
                        help="Force text-only mode (use captions, skip images)")
    
    args = parser.parse_args()
    
    max_samples = None if args.all else args.max_samples
    
    print(f"\n{'='*60}")
    print(f"AdMIRe 2.0 Full Evaluation - {args.language}")
    if args.text_only:
        print("MODE: Text-only (captions only, no images)")
    print(f"{'='*60}\n")
    
    run_full_evaluation(
        language=args.language,
        subset=args.subset,
        max_samples=max_samples,
        output_dir=args.output_dir,
        text_only=args.text_only
    )


if __name__ == "__main__":
    main()

