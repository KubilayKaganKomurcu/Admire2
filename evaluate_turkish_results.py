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
        
        # Simple comparison: first 2 elements of ranking vs gold
        # Predicted ranking: [5, 2, 1, 3, 4] → first two = [5, 2]
        # Gold: [5, 2]
        # Compare directly!
        pred_first2 = pred_ranking[:2]
        gold_first2 = gold[:2]
        
        pos1_match = pred_first2[0] == gold_first2[0]
        pos2_match = pred_first2[1] == gold_first2[1]
        
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
        print(f"  Pred first 2:      {pred_first2}")
        print(f"  Gold first 2:      {gold_first2}")
        print(f"  Position 1: Pred={pred_first2[0]}, Gold={gold_first2[0]} → {'✓' if pos1_match else '✗'}")
        print(f"  Position 2: Pred={pred_first2[1]}, Gold={gold_first2[1]} → {'✓' if pos2_match else '✗'}")
        
        # Show per-engine predictions
        engine_results = result.get('engine_results', {})
        print(f"\n  Per-engine predictions:")
        for engine_name, eng_result in engine_results.items():
            if 'ranking' in eng_result:
                eng_ranking = eng_result['ranking']
                eng_first2 = eng_ranking[:2]
                eng_pos1_match = eng_first2[0] == gold_first2[0]
                eng_pos2_match = eng_first2[1] == gold_first2[1]
                print(f"    {engine_name}: first2={eng_first2} → Pos1:{'✓' if eng_pos1_match else '✗'} Pos2:{'✓' if eng_pos2_match else '✗'}")
    
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

