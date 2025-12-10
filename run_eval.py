"""
AdMIRe 2.0 Evaluation Script
============================
Run evaluation on real data using the reorganized dataset structure.

Usage:
    python run_eval.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AdMIRe2Config, get_config
from data_loader import AdMIReItem
from engines import MIRAEngine, DAALFTEngine, CTYUNLiteEngine
from evaluation.ensemble import EnsembleAggregator, EnsembleResult
from evaluation.metrics import AdMIReEvaluator


def load_turkish_data(
    base_path: str = "data/admire2",
    max_samples: Optional[int] = None
) -> List[AdMIReItem]:
    """
    Load Turkish data from the reorganized dataset structure.
    
    Based on README_reorganization.md:
    - TSV file: data/admire2/TSVs/submission_Turkish.tsv
    - Images: data/admire2/languages/Turkish/<compound>/
    
    Args:
        base_path: Path to the admire2 data folder
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of AdMIReItem objects
    """
    base = Path(base_path)
    tsv_file = base / "TSVs" / "submission_Turkish.tsv"
    images_base = base / "languages" / "Turkish"
    
    if not tsv_file.exists():
        print(f"Error: TSV file not found at {tsv_file}")
        print("Make sure you have the data in the correct structure:")
        print("  data/admire2/TSVs/submission_Turkish.tsv")
        print("  data/admire2/languages/Turkish/<idiom_folders>/")
        return []
    
    # Read TSV
    df = pd.read_csv(tsv_file, sep='\t')
    
    print(f"Found {len(df)} items in Turkish dataset")
    print(f"Columns: {list(df.columns)}")
    
    # Limit samples if requested
    if max_samples:
        df = df.head(max_samples)
        print(f"Loading first {max_samples} samples for evaluation")
    
    items = []
    for idx, row in df.iterrows():
        compound = row["compound"]
        
        # Extract image names and captions (5 images per item)
        image_names = []
        image_captions = []
        for i in range(1, 6):
            name_col = f"image{i}_name"
            caption_col = f"image{i}_caption"
            
            if name_col in df.columns:
                image_names.append(str(row[name_col]) if pd.notna(row[name_col]) else f"image{i}.png")
            else:
                image_names.append(f"image{i}.png")
            
            if caption_col in df.columns:
                image_captions.append(str(row[caption_col]) if pd.notna(row[caption_col]) else "")
            else:
                image_captions.append("")
        
        # Build full image paths
        compound_folder = images_base / compound
        image_paths = [str(compound_folder / name) for name in image_names]
        
        # Parse expected order if present
        expected_order = None
        if "expected_order" in df.columns and pd.notna(row.get("expected_order")):
            order_str = str(row["expected_order"])
            order_str = order_str.strip("[]")
            expected_order = [s.strip().strip("'\"") for s in order_str.split(",")]
        
        # Get sentence type
        sentence_type = None
        if "sentence_type" in df.columns and pd.notna(row.get("sentence_type")):
            sentence_type = str(row["sentence_type"]).lower()
        
        # Get sentence
        sentence = str(row["sentence"]) if "sentence" in df.columns and pd.notna(row.get("sentence")) else ""
        
        item = AdMIReItem(
            compound=compound,
            sentence=sentence,
            image_names=image_names,
            image_captions=image_captions,
            image_paths=image_paths,
            sentence_type=sentence_type,
            expected_order=expected_order,
            subset=str(row.get("subset", "unknown")),
            language="tr",
            item_id=f"turkish_{idx}"
        )
        items.append(item)
    
    return items


def run_evaluation(
    items: List[AdMIReItem],
    config: Optional[AdMIRe2Config] = None,
    use_ensemble: bool = True,
    verbose: bool = True
):
    """
    Run evaluation on the provided items.
    
    Args:
        items: List of AdMIReItem to evaluate
        config: Configuration (uses default if None)
        use_ensemble: Whether to use ensemble or single engine
        verbose: Print detailed output
    """
    config = config or get_config()
    
    # Initialize engines
    engines = {}
    print("\n📦 Initializing engines...")
    
    if "mira" in config.ensemble.enabled_engines:
        engines["MIRA"] = MIRAEngine(config)
        print("  ✓ MIRA engine loaded")
    
    if "daalft" in config.ensemble.enabled_engines:
        engines["DAALFT"] = DAALFTEngine(config)
        print("  ✓ DAALFT engine loaded")
    
    if "ctyun_lite" in config.ensemble.enabled_engines:
        engines["CTYUN-Lite"] = CTYUNLiteEngine(config)
        print("  ✓ CTYUN-Lite engine loaded")
    
    # Initialize aggregator and evaluator
    aggregator = EnsembleAggregator(config.ensemble)
    evaluator = AdMIReEvaluator()
    
    print(f"\n🔍 Evaluating {len(items)} Turkish idiom items...\n")
    print("=" * 70)
    
    for i, item in enumerate(items, 1):
        print(f"\n{'='*70}")
        print(f"📌 Item {i}/{len(items)}: '{item.compound}'")
        print(f"   Sentence: {item.sentence[:80]}..." if len(item.sentence) > 80 else f"   Sentence: {item.sentence}")
        if item.sentence_type:
            print(f"   Gold type: {item.sentence_type}")
        print("-" * 70)
        
        # Run each engine
        engine_results = {}
        for name, engine in engines.items():
            if verbose:
                print(f"   Running {name}...")
            try:
                result = engine.rank_images(item)
                engine_results[name] = result
                if verbose:
                    print(f"     → Type: {result.sentence_type}, Ranking: {result.ranking}")
            except Exception as e:
                print(f"     ⚠ Error in {name}: {e}")
        
        if not engine_results:
            print("   ❌ No engine results, skipping...")
            continue
        
        # Aggregate results
        if use_ensemble and len(engine_results) > 1:
            ensemble_result = aggregator.aggregate(engine_results)
        else:
            # Use first engine's result
            first_result = list(engine_results.values())[0]
            ensemble_result = EnsembleResult(
                ranking=first_result.ranking,
                sentence_type=first_result.sentence_type,
                ranking_confidence=first_result.ranking_confidence,
                sentence_type_confidence=first_result.sentence_type_confidence,
                engine_results=engine_results,
                aggregation_method="single",
                agreement_score=1.0
            )
        
        # Display results
        print(f"\n📊 Ensemble Result:")
        print(f"   Sentence type: {ensemble_result.sentence_type} (conf: {ensemble_result.sentence_type_confidence:.2f})")
        print(f"   Ranking: {ensemble_result.ranking}")
        print(f"   Agreement score: {ensemble_result.agreement_score:.2f}")
        
        # Compare with gold if available
        if item.expected_order:
            ordered_pred = ensemble_result.to_ordered_images(item.image_names)
            print(f"\n🎯 Comparison with Gold:")
            print(f"   Gold order:      {item.expected_order}")
            print(f"   Predicted order: {ordered_pred}")
            
            matches = sum(1 for a, b in zip(item.expected_order, ordered_pred) if a == b)
            print(f"   Exact matches: {matches}/5")
            
            # Add to evaluator
            evaluator.add_prediction(
                predicted_ranking=ensemble_result.ranking,
                gold_ranking=item.expected_order,
                image_names=item.image_names,
                predicted_type=ensemble_result.sentence_type,
                gold_type=item.sentence_type
            )
    
    # Compute and display final metrics
    print("\n" + "=" * 70)
    print("📈 FINAL EVALUATION METRICS")
    print("=" * 70)
    
    metrics = evaluator.compute_metrics()
    print(metrics)
    
    return metrics


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🇹🇷 AdMIRe 2.0 - Turkish Language Evaluation")
    print("=" * 70)
    
    # Load Turkish data (only a few samples for demo)
    items = load_turkish_data(
        base_path="data/admire2",
        max_samples=3  # Limit to 3 examples for testing
    )
    
    if not items:
        print("\n❌ No data loaded. Please ensure the data is in place:")
        print("   1. data/admire2/TSVs/submission_Turkish.tsv")
        print("   2. data/admire2/languages/Turkish/<idiom_folders>/")
        return
    
    # Show sample data info
    print(f"\n📋 Sample data preview:")
    for i, item in enumerate(items[:3], 1):
        print(f"   {i}. {item.compound}")
        print(f"      Sentence: {item.sentence[:60]}..." if len(item.sentence) > 60 else f"      Sentence: {item.sentence}")
        print(f"      Images: {item.image_names}")
    
    # Run evaluation
    run_evaluation(items, verbose=True)
    
    print("\n" + "=" * 70)
    print("✅ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

