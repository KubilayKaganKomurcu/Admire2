# AdMIRe 2.0 - Category-Aware Image Ranking

A framework for **Multimodal Idiomaticity Representation** using GPT-5.1's vision capabilities.

## 🎯 Category Engine Approach

The Category Engine is built on a key insight: each compound expression in the dataset has **5 types of images**:

| Category | Description | Example ("bad apple") |
|----------|-------------|----------------------|
| **Literal** | Shows the exact words/objects | An actual rotten apple |
| **Literal-related** | Almost literal, partial match | A basket of apples |
| **Idiomatic** | Shows the figurative meaning | A troublemaker in a group |
| **Idiomatic-related** | Close to figurative meaning | Someone looking suspicious |
| **Distractor** | Superficially related but wrong | A peach (similar fruit, wrong context) |

### How It Works

**Step 1: Classify Sentence Type**

The engine first determines if the compound expression is used literally or idiomatically in the given sentence:

```
LITERAL = The actual, physical objects/words are being described
  - "bad apple" LITERAL → an actual rotten/bad apple fruit
  - "green fingers" LITERAL → fingers that are colored green

IDIOMATIC = A figurative/metaphorical meaning, not the actual objects
  - "bad apple" IDIOMATIC → a troublemaker, someone who corrupts others  
  - "green fingers" IDIOMATIC → skilled at gardening
```

**Step 2: Category-Aware Ranking**

Based on the classification, the ranking strategy changes:

**For LITERAL usage:**
- ✓ BEST (Rank 1): Images containing the actual words/objects → LITERAL
- ✓ GOOD (Rank 2): Images with only one word visible → LITERAL-RELATED
- ✗ AVOID (Rank 3+): Images without the words → IDIOMATIC/DISTRACTOR

**For IDIOMATIC usage:**
- ✓ BEST (Rank 1): Images showing the figurative meaning (NO literal objects) → IDIOMATIC
- ✓ GOOD (Rank 2): Images close to idiomatic meaning → IDIOMATIC-RELATED
- ✗ AVOID (Rank 3+): Images with literal objects → These are WRONG for idiomatic usage!

### Key Insight

The engine explicitly checks for the presence or absence of literal objects when ranking:
- Split compound into component words (e.g., "bad apple" → ["bad", "apple"])
- For literal usage: prioritize images where both words are visible
- For idiomatic usage: **penalize** images where literal objects appear

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install openai pandas numpy tqdm scipy
```

### 2. Configure API Key

Set your OpenAI API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-proj-your-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-proj-your-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-proj-your-key-here"
```

### 3. Generate Submissions

```bash
# Single language
python generate_submission.py --language Turkish --engine category

# All languages (except TR, KK, UZ, IG)
python run_all_submissions.py --engine category

# Text-only mode (captions only, no images)
python generate_submission.py --language Turkish --engine category --text-only
```

## 📁 Data Setup

Place your AdMIRe 2.0 data in the following structure:

```
Admire2/
└── data/
    ├── TSVs/
    │   └── submission_[LANG].tsv
    └── languages/
        └── [Language]/
            └── [compound_folders]/
                └── [image_files].png
```

## 📊 Output Format

### Submission TSV

```tsv
compound    sentence    expected_order
bad apple   He's a bad apple in the team.   ['img3.png', 'img1.png', 'img2.png', 'img5.png', 'img4.png']
```

### Ranking Array

The ranking `[3, 1, 2, 5, 4]` means:
- Position 1: Image 3 (best match)
- Position 2: Image 1
- Position 3: Image 2
- Position 4: Image 5
- Position 5: Image 4 (worst match)

## 📈 Evaluation Metrics

- **DCG** / **NDCG**: Discounted Cumulative Gain
- **Acc@1**: Top prediction accuracy
- **Acc@3**: Correct answer in top 3
- **MRR**: Mean Reciprocal Rank
- **Sentence Type Accuracy**: Idiomatic vs literal classification

## 🌐 Multi-language Support

The system supports zero-shot cross-lingual transfer:
- All prompts work with any language
- GPT-5.1 handles multilingual text naturally
- Works with both images and text captions

## 📝 Example Usage

```python
from config import get_config
from engines import CategoryEngine
from data_loader import AdMIReItem

# Initialize
config = get_config()
engine = CategoryEngine(config)

# Create an item
item = AdMIReItem(
    compound="bad apple",
    sentence="He's always been the bad apple in our team.",
    image_names=["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"],
    image_captions=["A rotten apple", "Group of people", ...],
    image_paths=["path/to/img1.png", ...],
    language="english"
)

# Get ranking
result = engine.rank_images(item)
print(f"Ranking: {result.ranking}")
print(f"Sentence Type: {result.sentence_type}")
```

## 🛠️ Troubleshooting

### API Errors
- Check your API key is set correctly
- Ensure you have GPT-5.1 API access
- For rate limits, the engine includes automatic retry logic

### Missing Images
- System automatically falls back to **caption-only** mode
- This uses the same category-aware logic with text descriptions

---

*AdMIRe 2.0 Category Engine - Built for SemEval-2025 Task 1*
