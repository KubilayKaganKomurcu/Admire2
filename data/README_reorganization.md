# Dataset Structure

## Overview

This dataset contains idiom data for multiple languages with images and TSV metadata files.

---

## Folder Structure

```
admire2/
├── languages/
│   ├── EN/                    # 100 idiom folders with images
│   ├── PT/                    # 55 idiom folders with images
│   ├── Chinese/
│   ├── Georgian/
│   ├── Greek/
│   └── ... (other languages)
│
└── TSVs/
    ├── EN_subtask_a.tsv       # English - 100 idioms
    ├── PT_subtask_a.tsv       # Portuguese - 55 idioms
    ├── EN_subtask_a_dev.xlsx
    ├── EN_subtask_a_train.xlsx
    └── submission_*.tsv       # Other language submissions
```

---

## Languages Folder

Each language folder contains idiom subfolders with PNG images.

| Language | Idiom Count | Structure |
|----------|-------------|-----------|
| EN (English) | 100 | Flat - idiom folders in root |
| PT (Portuguese) | 55 | Flat - idiom folders in root |
| Chinese | varies | Flat - idiom folders in root |
| Georgian | varies | Flat - idiom folders in root |
| ... | ... | ... |

### Example Structure
```
languages/EN/
├── acid test/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── act of god/
├── best man/
└── ... (100 total idiom folders)
```

---

## TSVs Folder

### EN and PT Data Files

| File | Description | Idioms |
|------|-------------|--------|
| `EN_subtask_a.tsv` | English merged data | 100 |
| `PT_subtask_a.tsv` | Portuguese merged data | 55 |
| `EN_subtask_a_dev.xlsx` | English dev (Excel) | - |
| `EN_subtask_a_train.xlsx` | English train (Excel) | - |

> **Note:** The TSV files contain a `subset` column (Dev/Test/Train) to identify the original data split.

### Other Language Submissions

- `submission_Chinese.tsv`
- `submission_Georgian.tsv`
- `submission_Greek.tsv`
- `submission_Igbo.tsv`
- `submission_Kazakh.tsv`
- `submission_Norwegian.tsv`
- `submission_Portuguese-Brazil.tsv`
- `submission_Portuguese-Portugal.tsv`
- `submission_Russian.tsv`
- `submission_Serbian.tsv`
- `submission_Slovak.tsv`
- `submission_Slovenian.tsv`
- `submission_Spanish-Ecuador.tsv`
- `submission_Turkish.tsv`
- `submission_Uzbek.tsv`

---

## TSV File Format

The TSV files contain the following columns:

| Column | Description |
|--------|-------------|
| `compound` | Idiom name |
| `subset` | Data split (Dev/Test/Train) |
| `sentence_type` | idiomatic or literal |
| `sentence` | Example sentence |
| `expected_order` | Expected image order |
| `image1_name` | First image filename |
| `image1_caption` | First image caption |
| ... | (up to 5 images with captions) |
