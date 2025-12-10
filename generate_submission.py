"""
Generate Submission File for AdMIRe 2.0
=======================================
Creates a TSV submission file in the required format.

Usage:
    python generate_submission.py --language Turkish --engine category
    python generate_submission.py --language Turkish --engine category --text-only
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config, CONFIG
from data_loader import AdMIReItem
from engines import MIRAEngine, DAALFTEngine, CTYUNLiteEngine, CategoryEngine
from evaluation.ensemble import EnsembleAggregator

# Language code mapping
LANGUAGE_CODES = {
    "Turkish": "TR",
    "Chinese": "ZH",
    "Georgian": "KA",
    "Greek": "EL",
    "Igbo": "IG",
    "Kazakh": "KK",
    "Norwegian": "NO",
    "Portuguese-Brazil": "PT-BR",
    "Portuguese-Portugal": "PT-PT",
    "Russian": "RU",
    "Serbian": "SR",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Spanish-Ecuador": "ES-EC",
    "Uzbek": "UZ",
}


def load_language_data(language: str, base_path: str = "data") -> List[AdMIReItem]:
    """Load all data for a language."""
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
    
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"Found {len(df)} items in {language} dataset")
    
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
        
        # Get sentence
        sentence = str(row.get("sentence", ""))
        
        item = AdMIReItem(
            compound=compound,
            sentence=sentence,
            image_names=image_names,
            image_captions=image_captions,
            image_paths=image_paths,
            sentence_type=None,
            expected_order=None,
            subset=str(row.get("subset", "unknown")),
            language=language.lower(),
            item_id=f"{language}_{idx}"
        )
        items.append(item)
    
    return items


def ranking_to_ordered_names(ranking: List[int], image_names: List[str]) -> List[str]:
    """
    Convert ranking array to ordered list of image names.
    
    ranking = [3, 1, 4, 2, 5] means:
    - Image 1 has rank 3
    - Image 2 has rank 1 (best)
    - Image 3 has rank 4
    - Image 4 has rank 2
    - Image 5 has rank 5 (worst)
    
    Wait, based on user's clarification, ranking[0], ranking[1] are the first two
    positions directly. So ranking = [3, 4, 1, 2, 5] means image 3 is 1st, image 4 is 2nd, etc.
    
    Actually, let me re-read. The ranking array's first two elements ARE the image numbers
    that should be in positions 1 and 2. So ranking = [5, 2, 1, 3, 4] means:
    - Position 1: Image 5
    - Position 2: Image 2
    - etc.
    
    So to get ordered names, we just map the ranking numbers to image names.
    ranking[i] = image number at position i+1
    """
    ordered_names = []
    for img_num in ranking:
        if 1 <= img_num <= len(image_names):
            ordered_names.append(image_names[img_num - 1])
        else:
            ordered_names.append(image_names[0])  # Fallback
    return ordered_names


def generate_submission(
    language: str = "Turkish",
    engine_name: str = "category",
    text_only: bool = False,
    output_dir: str = "submissions"
):
    """Generate submission TSV file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get language code
    lang_code = LANGUAGE_CODES.get(language, language[:2].upper())
    output_file = os.path.join(output_dir, f"submission_{lang_code}.tsv")
    
    print(f"\n{'=' * 60}")
    print(f"Generating Submission for {language} ({lang_code})")
    print(f"Engine: {engine_name}")
    print(f"Mode: {'Text-only' if text_only else 'With images'}")
    print(f"{'=' * 60}\n")
    
    # Load data
    items = load_language_data(language)
    if not items:
        print("No data loaded!")
        return
    
    # Initialize engine
    config = get_config()
    
    if engine_name == "category":
        engine = CategoryEngine(config)
    elif engine_name == "mira":
        engine = MIRAEngine(config)
    elif engine_name == "daalft":
        engine = DAALFTEngine(config)
    elif engine_name == "ctyun_lite":
        engine = CTYUNLiteEngine(config)
    else:
        print(f"Unknown engine: {engine_name}")
        return
    
    print(f"Engine loaded: {engine.name}")
    
    # Process each item
    results = []
    
    for i, item in enumerate(items, 1):
        print(f"\rProcessing {i}/{len(items)}: {item.compound[:30]}...", end="", flush=True)
        
        # Force text-only mode if requested
        if text_only:
            item.image_paths = []
        
        try:
            # Get prediction
            result = engine.rank_images(item)
            ranking = result.ranking
            
            # Convert ranking to ordered image names
            ordered_names = ranking_to_ordered_names(ranking, item.image_names)
            
            # Format as required: ['img1.png', 'img2.png', ...]
            expected_order_str = str(ordered_names)
            
            results.append({
                "compound": item.compound,
                "expected_order": expected_order_str
            })
            
        except Exception as e:
            print(f"\nError processing {item.compound}: {e}")
            # Use default order on error
            results.append({
                "compound": item.compound,
                "expected_order": str(item.image_names)
            })
    
    print(f"\n\nProcessed {len(results)} items")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Submission saved to: {output_file}")
    print(f"{'=' * 60}")
    
    # Show sample
    print(f"\nSample output (first 3 rows):")
    print(df.head(3).to_string())
    
    return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AdMIRe 2.0 Submission")
    parser.add_argument("--language", default="Turkish",
                        help="Language to process (Turkish, Chinese, Greek, etc.)")
    parser.add_argument("--engine", default="category",
                        choices=["category", "mira", "daalft", "ctyun_lite"],
                        help="Engine to use")
    parser.add_argument("--text-only", action="store_true", dest="text_only",
                        help="Force text-only mode (captions only)")
    parser.add_argument("--output_dir", default="submissions",
                        help="Output directory for submission files")
    
    args = parser.parse_args()
    
    generate_submission(
        language=args.language,
        engine_name=args.engine,
        text_only=args.text_only,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

