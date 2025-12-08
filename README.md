# AdMIRe 2.0 Ensemble System

A unified, training-free framework for **Multimodal Idiomaticity Representation** that combines three state-of-the-art approaches:

| Engine | Approach | Key Technique |
|--------|----------|---------------|
| **MIRA** | Self-consistency + Multi-step | Multiple samples → Borda aggregation |
| **DAALFT** | Detect → Explain → Rank | Explicit reasoning chain |
| **CTYUN-Lite** | Caption-based direct ranking | Text-only, efficient |

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

**Permanent (add to your shell profile or .bashrc):**
```bash
echo 'export OPENAI_API_KEY="sk-proj-your-key-here"' >> ~/.bashrc
```

### 3. Run Demo

```bash
python run_demo.py
```

Or with more options:

```bash
python main.py --mode demo      # Demo with mock data
python main.py --mode evaluate  # Evaluate on dataset
python main.py --mode predict   # Generate submission
```

## 📁 Data Setup

Place your AdMIRe 2.0 data in the following structure:

```
Admire2/
└── data/
    ├── subtask_a/
    │   ├── english/
    │   │   ├── subtask_a_train.tsv
    │   │   ├── subtask_a_dev.tsv
    │   │   └── [compound_folders]/
    │   │       └── [image_files].png
    │   └── portuguese/
    │       └── ...
    └── subtask_b/
        ├── subtask_b_train.tsv
        └── [compound_folders]/
```

## 🔧 Configuration

All settings are in `config.py`:

```python
# Engine selection
enabled_engines: List[str] = ["mira", "daalft", "ctyun_lite"]

# MIRA settings
mira_num_samples: int = 3  # More samples = more robust, slower
mira_temperature: float = 0.7  # Diversity for self-consistency

# DAALFT settings
daalft_include_explanation: bool = True  # Generate meaning explanations
daalft_chain_of_thought: bool = True  # Use CoT prompting

# Ensemble weights
engine_weights: dict = {
    "mira": 1.0,
    "daalft": 1.0,
    "ctyun_lite": 0.8  # Slightly lower for text-only
}
```

## 📊 Outputs

### Ranking Format

For each item, the system outputs:
- **ranking**: `[1, 3, 2, 5, 4]` where position `i` = rank of image `i`
- **sentence_type**: `"idiomatic"` or `"literal"`
- **confidence scores** for both predictions

### Submission Format

```tsv
compound	predicted_order	sentence_type
green fingers	img1.png,img3.png,img2.png,img5.png,img4.png	idiomatic
...
```

## 🏗️ Architecture

```
Input: (compound, sentence, images/captions)
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  MIRA    │  │  DAALFT  │  │  CTYUN   │
│  Engine  │  │  Engine  │  │  Lite    │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     └─────────────┼─────────────┘
                   ▼
          Weighted Borda Count
                   │
                   ▼
        Final: (ranking, type)
```

## 📈 Evaluation Metrics

- **DCG** / **NDCG**: Discounted Cumulative Gain
- **Acc@1**: Top prediction accuracy
- **Acc@3**: Correct answer in top 3
- **MRR**: Mean Reciprocal Rank
- **Sentence Type Accuracy**: Idiomatic vs literal classification

## 🔍 Engine Details

### MIRA (Multimodal Idiom Recognition and Alignment)
- Generates **multiple rankings** with temperature sampling
- Aggregates via **Borda count** for stability
- Best for: **robustness**, handling ambiguous cases

### DAALFT (Detect-Analyze-Align-Rank)
- **Step 1**: Detect if expression is idiomatic/literal
- **Step 2**: Generate meaning explanation
- **Step 3**: Rank with context
- Best for: **interpretability**, explicit reasoning

### CTYUN-Lite (Caption-based Ranking)
- Uses **text captions only** (no image processing)
- Direct, efficient ranking
- Best for: **speed**, text-only scenarios

## 🌐 Multi-language Support

The system is designed for **zero-shot cross-lingual transfer**:
- All prompts work with any language
- GPT-4o handles multilingual text naturally
- Portuguese captions supported when available

## 📝 Example Usage

```python
from main import AdMIRe2System
from data_loader import create_mock_data

# Initialize system
system = AdMIRe2System()

# Get mock data
items = create_mock_data()

# Single prediction
result = system.predict_single(items[0])
print(f"Ranking: {result.ranking}")
print(f"Type: {result.sentence_type}")

# Batch prediction with evaluation
metrics = system.evaluate(language="english", split="dev")
print(metrics)
```

## 🛠️ Troubleshooting

### API Errors
- Check your API key in `config.py`
- Ensure you have GPT-5 API access
- For rate limits, increase `retry_delay` in config

## 💰 Cost Estimation

Using GPT-5 series (much cheaper than GPT-4!):

| Model | Input/1M | Output/1M | Used For |
|-------|----------|-----------|----------|
| GPT-5 | $1.25 | $10.00 | Vision ranking |
| GPT-5 Mini | $0.25 | $2.00 | Text classification |
| GPT-5 Nano | $0.05 | $0.40 | (optional) Ultra-cheap |

**Estimated cost per item**: ~$0.01-0.03 (depending on engines used)
**100 items**: ~$1-3

### Missing Images
- System automatically falls back to **caption-only** mode
- This is the expected behavior for text-only evaluation

### Memory Issues
- Reduce `mira_num_samples` to 1 or 2
- Use `ctyun_lite` engine only for minimal memory

## 📚 References

- **MIRA**: PALI-NLP at SemEval-2025 Task 1
- **DAALFT**: Multi-step zero-shot reasoning for idiom ranking
- **CTYUN**: Learning-to-rank with Qwen (adapted for zero-shot)

---

*AdMIRe 2.0 Ensemble System - Built for SemEval-2025 Task 1*

