"""
Add sentence column to existing submission files
================================================
Merges sentence data from original TSVs into submission files.

Usage:
    python add_sentence_to_submission.py
"""

import pandas as pd
from pathlib import Path

# Language mappings
LANGUAGES = {
    "TR": "Turkish",
    "IG": "Igbo",
    "KK": "Kazakh",
    "UZ": "Uzbek",
}


def add_sentences_to_submission(lang_code: str, language_name: str):
    """Add sentence column to a submission file."""
    
    # Paths
    submission_file = Path(f"submissions/submission_{lang_code}.tsv")
    original_file = Path(f"data/TSVs/submission_{language_name}.tsv")
    output_file = Path(f"submissions/submission_{lang_code}_with_sentence.tsv")
    
    if not submission_file.exists():
        print(f"Submission file not found: {submission_file}")
        return
    
    if not original_file.exists():
        print(f"Original file not found: {original_file}")
        return
    
    # Read files
    submission_df = pd.read_csv(submission_file, sep='\t')
    original_df = pd.read_csv(original_file, sep='\t')
    
    print(f"\n{'='*60}")
    print(f"Processing {lang_code} ({language_name})")
    print(f"{'='*60}")
    print(f"Submission rows: {len(submission_df)}")
    print(f"Original rows: {len(original_df)}")
    
    # Check if they have same number of rows (should match by position)
    if len(submission_df) != len(original_df):
        print(f"WARNING: Row count mismatch!")
    
    # Add sentence column by position (assuming same order)
    if 'sentence' in original_df.columns:
        submission_df['sentence'] = original_df['sentence'].values[:len(submission_df)]
        
        # Reorder columns: compound, sentence, expected_order
        cols = ['compound', 'sentence', 'expected_order']
        submission_df = submission_df[cols]
        
        # Save
        submission_df.to_csv(output_file, sep='\t', index=False)
        print(f"Saved to: {output_file}")
        
        # Show sample
        print(f"\nSample (first 3 rows):")
        print(submission_df.head(3).to_string())
    else:
        print(f"No 'sentence' column in original file!")
        print(f"Available columns: {list(original_df.columns)}")


def main():
    print("Adding sentence column to submission files...")
    
    for lang_code, language_name in LANGUAGES.items():
        add_sentences_to_submission(lang_code, language_name)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

