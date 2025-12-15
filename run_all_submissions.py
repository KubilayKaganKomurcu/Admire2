"""
Run Submission Generation for All Languages
============================================
Generates submission files for all languages except TR, KK, UZ, IG.

Usage:
    python run_all_submissions.py
    python run_all_submissions.py --engine category
"""

import subprocess
import sys
from datetime import datetime

# All languages from generate_submission.py
ALL_LANGUAGES = [
    "Turkish",        # TR
    "Chinese",        # ZH
    "Georgian",       # KA
    "Greek",          # EL
    "Igbo",           # IG
    "Kazakh",         # KK
    "Norwegian",      # NO
    "Portuguese-Brazil",   # PT-BR
    "Portuguese-Portugal", # PT-PT
    "Russian",        # RU
    "Serbian",        # SR
    "Slovak",         # SK
    "Slovenian",      # SL
    "Spanish-Ecuador", # ES-EC
    "Uzbek",          # UZ
]

# Languages to exclude (by code: TR, KK, UZ, IG)
EXCLUDED_LANGUAGES = [
    "Turkish",   # TR
    "Kazakh",    # KK
    "Uzbek",     # UZ
    "Igbo",      # IG
]

# Languages to process
LANGUAGES_TO_PROCESS = [lang for lang in ALL_LANGUAGES if lang not in EXCLUDED_LANGUAGES]


def run_submission_for_language(language: str, engine: str = "category"):
    """Run generate_submission.py for a single language."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {language}")
    print(f"{'=' * 60}")
    
    cmd = [
        sys.executable,
        "generate_submission.py",
        "--language", language,
        "--engine", engine
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {language}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run submission generation for all languages")
    parser.add_argument("--engine", default="category",
                        choices=["category", "mira", "daalft", "ctyun_lite"],
                        help="Engine to use for all languages")
    
    args = parser.parse_args()
    
    print(f"\n{'#' * 60}")
    print(f"# Batch Submission Generation")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Engine: {args.engine}")
    print(f"# Languages: {len(LANGUAGES_TO_PROCESS)}")
    print(f"{'#' * 60}")
    
    print(f"\nLanguages to process:")
    for lang in LANGUAGES_TO_PROCESS:
        print(f"  - {lang}")
    
    print(f"\nExcluded languages: {EXCLUDED_LANGUAGES}")
    
    # Track results
    successful = []
    failed = []
    
    for language in LANGUAGES_TO_PROCESS:
        success = run_submission_for_language(language, args.engine)
        if success:
            successful.append(language)
        else:
            failed.append(language)
    
    # Summary
    print(f"\n{'#' * 60}")
    print(f"# SUMMARY")
    print(f"{'#' * 60}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {len(successful)}/{len(LANGUAGES_TO_PROCESS)}")
    
    if successful:
        print(f"\nSuccessfully processed:")
        for lang in successful:
            print(f"  ✓ {lang}")
    
    if failed:
        print(f"\nFailed:")
        for lang in failed:
            print(f"  ✗ {lang}")
    
    print(f"\n{'#' * 60}")


if __name__ == "__main__":
    main()

