# Category Engine

A category-aware ranking engine for the AdMIRe idiom-image matching task.

## Core Concept

Each idiom in the dataset has **5 candidate images** that fall into specific categories:

| Category | Description | Example for "bad apple" |
|----------|-------------|-------------------------|
| **Literal** | Shows exact words/objects | A rotten apple fruit |
| **Literal-related** | Almost literal, partial match | A basket of apples |
| **Idiomatic** | Shows figurative meaning | A troublemaker in a group |
| **Idiomatic-related** | Close to figurative | A shady/suspicious person |
| **Distractor** | Superficially related but wrong | A peach (similar fruit, wrong meaning) |

## Strategy

The engine uses a **two-step approach**:

### Step 1: Classify Usage Type
Determine if the idiom is used **literally** or **idiomatically** in the sentence.

```
"The bad apple spoiled the whole barrel" → LITERAL (actual fruit)
"He's the bad apple of the team" → IDIOMATIC (troublemaker)
```

### Step 2: Category-Aware Ranking

**For LITERAL usage:**
```
BEST (Rank 1-2):  Images with actual objects (apple + bad/rotten)
GOOD (Rank 3):    Images with partial match (just apples)
AVOID (Rank 4-5): Images showing figurative meaning
```

**For IDIOMATIC usage:**
```
BEST (Rank 1-2):  Images showing figurative meaning (NO literal objects)
GOOD (Rank 3):    Images close to figurative meaning
AVOID (Rank 4-5): Images with literal objects (apple visible = WRONG)
```

## Key Insight

The critical realization is that **literal images are WRONG for idiomatic usage** and vice versa. The engine explicitly:

1. Checks if literal objects (the actual words) are visible in the image
2. Uses this as a **negative signal** for idiomatic usage
3. Uses this as a **positive signal** for literal usage

## Usage

```python
from engines import CategoryEngine
from config import get_config

engine = CategoryEngine(get_config())
result = engine.rank_images(item)

print(result.ranking)        # [2, 4, 1, 5, 3] = ranks for each image
print(result.sentence_type)  # "idiomatic" or "literal"
```

## How It Works

### Input
- `compound`: The idiom (e.g., "bad apple")
- `sentence`: Context sentence
- `image_captions`: Descriptions of 5 candidate images

### Process
1. **Parse compound** into words: `["bad", "apple"]`
2. **Classify usage** from sentence context
3. **Analyze each image/caption**:
   - Check for literal word matches
   - Evaluate idiomatic meaning alignment
   - Identify distractors
4. **Rank based on usage type**

### Output
- `ranking`: List of 5 integers (rank for each image position)
- `sentence_type`: "literal" or "idiomatic"
- `confidence`: Confidence score

## Example

**Compound:** "çürük elma" (Turkish for "bad apple")

**Sentence:** "Sosyal medyada paylaşılan görüntüde, çürük elmaların meyve suyu yapılması için fabrikalara gönderildiği..."

**Analysis:**
- Usage: **LITERAL** (talking about actual rotten apples being sent to factories)
- Strategy: Find images showing actual rotten/bad apples

**Ranking logic:**
1. Image with rotten apple → Rank 1 (LITERAL match)
2. Image with basket of apples → Rank 2 (LITERAL-RELATED)
3. Image of troublemakers → Rank 5 (IDIOMATIC - wrong for this context)

