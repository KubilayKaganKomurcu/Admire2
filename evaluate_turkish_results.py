"""
Evaluate Turkish Results
========================
Compare predictions with gold labels (only first 2 positions matter).

Usage:
    python evaluate_turkish_results.py eval_results/eval_Turkish_textonly_20251210_174001.json
"""

import json
import sys
from pathlib import Path

# Gold labels for Turkish (first 5 items)
# Format: [position1, position2, x, x, x] where x = don't care
# These are image NUMBERS (1-5) that should be in positions 1 and 2
TURKISH_GOLD = {
    0: [3, 4],  # Item 1: Image 3 should be 1st, Image 4 should be 2nd
    1: [5, 2],  # Item 2: Image 5 should be 1st, Image 2 should be 2nd
    2: [2, 4],  # Item 3
    3: [1, 2],  # Item 4
    4: [2, 1],  # Item 5
}


def ranking_to_order(ranking):
    """
    Convert ranking array to ordered list.
    ranking[i] = rank of image i+1
    Returns: list of image numbers in order (best first)
    """
    # Create (rank, image_number) pairs
    paired = [(rank, img_num) for img_num, rank in enumerate(ranking, 1)]
    paired.sort(key=lambda x: x[0])
    return [img_num for _, img_num in paired]


def evaluate_results(json_file: str):
    """Evaluate results from JSON file."""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    print("=" * 70)
    print("TURKISH EVALUATION - First 2 Positions Only")
    print("=" * 70)
    
    total_items = min(len(results), len(TURKISH_GOLD))
    pos1_correct = 0
    pos2_correct = 0
    both_correct = 0
    
    for i in range(total_items):
        result = results[i]
        gold = TURKISH_GOLD[i]
        
        compound = result.get('compound', f'Item {i+1}')
        
        # Get ensemble prediction
        ensemble = result.get('ensemble_result', {})
        pred_ranking = ensemble.get('ranking', [1, 2, 3, 4, 5])
        
        # Convert ranking to order (which image is 1st, 2nd, etc.)
        pred_order = ranking_to_order(pred_ranking)
        
        # Gold labels are in ORDER format (which images should be 1st, 2nd)
        gold_pos1_img = gold[0]  # Image number that should be 1st
        gold_pos2_img = gold[1]  # Image number that should be 2nd
        
        # Check ranking directly: gold images should have ranks 1 and 2
        # ranking[i-1] = rank of image i
        pred_rank_img1 = pred_ranking[gold_pos1_img - 1] if gold_pos1_img <= len(pred_ranking) else None
        pred_rank_img2 = pred_ranking[gold_pos2_img - 1] if gold_pos2_img <= len(pred_ranking) else None
        
        # Gold images should have ranks 1 and 2
        pos1_match = pred_rank_img1 == 1  # Gold image 1 should have rank 1
        pos2_match = pred_rank_img2 == 2  # Gold image 2 should have rank 2
        
        # Also show order format for comparison
        pred_pos1 = pred_order[0] if len(pred_order) > 0 else None
        pred_pos2 = pred_order[1] if len(pred_order) > 1 else None
        
        if pos1_match:
            pos1_correct += 1
        if pos2_match:
            pos2_correct += 1
        if pos1_match and pos2_match:
            both_correct += 1
        
        # Print per-item results
        print(f"\n{'─' * 70}")
        print(f"Item {i+1}: '{compound}'")
        print(f"  Predicted ranking: {pred_ranking}")
        print(f"  Predicted order:   {pred_order}")
        print(f"  Gold (first 2 images): {gold} (Image {gold_pos1_img} should be 1st, Image {gold_pos2_img} should be 2nd)")
        print(f"  Ranking check:")
        print(f"    Image {gold_pos1_img} rank: Pred={pred_rank_img1}, Gold=1 → {'✓' if pos1_match else '✗'}")
        print(f"    Image {gold_pos2_img} rank: Pred={pred_rank_img2}, Gold=2 → {'✓' if pos2_match else '✗'}")
        print(f"  Order check (for reference):")
        print(f"    Position 1: Pred={pred_pos1}, Gold={gold_pos1_img} → {'✓' if pred_pos1 == gold_pos1_img else '✗'}")
        print(f"    Position 2: Pred={pred_pos2}, Gold={gold_pos2_img} → {'✓' if pred_pos2 == gold_pos2_img else '✗'}")
        
        # Show per-engine predictions
        engine_results = result.get('engine_results', {})
        print(f"\n  Per-engine predictions:")
        for engine_name, eng_result in engine_results.items():
            if 'ranking' in eng_result:
                eng_ranking = eng_result['ranking']
                eng_order = ranking_to_order(eng_ranking)
                eng_rank_img1 = eng_ranking[gold_pos1_img - 1] if gold_pos1_img <= len(eng_ranking) else None
                eng_rank_img2 = eng_ranking[gold_pos2_img - 1] if gold_pos2_img <= len(eng_ranking) else None
                eng_pos1_match = eng_rank_img1 == 1
                eng_pos2_match = eng_rank_img2 == 2
                print(f"    {engine_name}: ranking={eng_ranking}, order={eng_order[:2]}")
                print(f"      Image {gold_pos1_img} rank={eng_rank_img1}, Image {gold_pos2_img} rank={eng_rank_img2} → Pos1:{'✓' if eng_pos1_match else '✗'} Pos2:{'✓' if eng_pos2_match else '✗'}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total items evaluated: {total_items}")
    print(f"Position 1 accuracy: {pos1_correct}/{total_items} = {100*pos1_correct/total_items:.1f}%")
    print(f"Position 2 accuracy: {pos2_correct}/{total_items} = {100*pos2_correct/total_items:.1f}%")
    print(f"Both positions correct: {both_correct}/{total_items} = {100*both_correct/total_items:.1f}%")
    print(f"Average (Pos1 + Pos2): {100*(pos1_correct + pos2_correct)/(2*total_items):.1f}%")
    
    return {
        "total": total_items,
        "pos1_acc": pos1_correct / total_items,
        "pos2_acc": pos2_correct / total_items,
        "both_acc": both_correct / total_items
    }


def main():
    if len(sys.argv) < 2:
        # Try to find most recent Turkish eval file
        eval_dir = Path("eval_results")
        if eval_dir.exists():
            json_files = list(eval_dir.glob("eval_Turkish*.json"))
            if json_files:
                json_file = max(json_files, key=lambda x: x.stat().st_mtime)
                print(f"Using most recent file: {json_file}")
            else:
                print("Usage: python evaluate_turkish_results.py <json_file>")
                print("No Turkish eval files found in eval_results/")
                return
        else:
            print("Usage: python evaluate_turkish_results.py <json_file>")
            return
    else:
        json_file = sys.argv[1]
    
    evaluate_results(json_file)


if __name__ == "__main__":
    main()

