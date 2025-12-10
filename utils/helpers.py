"""
AdMIRe 2.0 Helper Utilities
===========================
Common utility functions used across the system.
"""

import base64
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string for API calls."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the media type of an image based on extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


def parse_ranking_from_response(response: str) -> List[int]:
    """
    Extract ranking from LLM response.
    Handles various formats like:
    - "Ranking: 3, 1, 5, 2, 4"
    - "1. image3, 2. image1, ..."
    - "[3, 1, 5, 2, 4]"
    """
    # Try JSON array format
    json_match = re.search(r'\[[\d,\s]+\]', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try "Ranking: X, Y, Z" format
    ranking_match = re.search(r'[Rr]anking[:\s]+([1-5][\s,]+[1-5][\s,]+[1-5][\s,]+[1-5][\s,]+[1-5])', response)
    if ranking_match:
        numbers = re.findall(r'[1-5]', ranking_match.group(1))
        return [int(n) for n in numbers]
    
    # Try finding any sequence of 5 distinct numbers 1-5
    all_numbers = re.findall(r'[1-5]', response)
    if len(all_numbers) >= 5:
        # Take first 5 unique numbers
        seen = set()
        result = []
        for n in all_numbers:
            if n not in seen:
                seen.add(n)
                result.append(int(n))
            if len(result) == 5:
                return result
    
    # Default fallback (using reverse order to detect parsing failures)
    print(f"  ⚠️ PARSE FALLBACK: Could not parse ranking from response")
    return [5, 4, 3, 2, 1]


def parse_sentence_type_from_response(response: str) -> Tuple[str, float]:
    """
    Extract sentence type (idiomatic/literal) and confidence from response.
    Returns: (sentence_type, confidence)
    """
    response_lower = response.lower()
    
    # Look for explicit classification
    if "idiomatic" in response_lower:
        # Try to extract confidence
        conf_match = re.search(r'(?:confidence|probability)[:\s]*([\d.]+)', response_lower)
        confidence = float(conf_match.group(1)) if conf_match else 0.8
        return "idiomatic", min(confidence, 1.0)
    
    elif "literal" in response_lower:
        conf_match = re.search(r'(?:confidence|probability)[:\s]*([\d.]+)', response_lower)
        confidence = float(conf_match.group(1)) if conf_match else 0.8
        return "literal", min(confidence, 1.0)
    
    # Default
    return "idiomatic", 0.5


def parse_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def normalize_ranking(ranking: List[int]) -> List[int]:
    """
    Normalize ranking to ensure it contains exactly [1,2,3,4,5] as a permutation.
    """
    if len(ranking) != 5:
        print(f"  ⚠️ NORMALIZE FALLBACK: Ranking length is {len(ranking)}, not 5. Ranking: {ranking}")
        return [5, 4, 3, 2, 1]
    
    # Check if valid permutation
    if set(ranking) == {1, 2, 3, 4, 5}:
        return ranking
    
    # If not valid, return default
    print(f"  ⚠️ NORMALIZE FALLBACK: Invalid permutation {ranking}")
    return [5, 4, 3, 2, 1]


def ranking_to_image_order(ranking: List[int], image_names: List[str]) -> List[str]:
    """
    Convert numeric ranking to ordered list of image names.
    ranking[i] = position of image i+1 in the final order
    """
    if len(ranking) != len(image_names):
        return image_names
    
    # Create pairs of (rank, image_name) and sort by rank
    paired = list(zip(ranking, image_names))
    paired.sort(key=lambda x: x[0])
    return [name for _, name in paired]


def calculate_dcg(ranking: List[int], relevance: List[float], k: int = 5) -> float:
    """
    Calculate Discounted Cumulative Gain.
    
    Args:
        ranking: Predicted ranking (1-indexed positions)
        relevance: Relevance scores for each item
        k: Number of top items to consider
    """
    import math
    
    dcg = 0.0
    for i in range(min(k, len(ranking))):
        # Get the item at position i in the ranking
        item_idx = ranking[i] - 1 if ranking[i] <= len(relevance) else 0
        rel = relevance[item_idx] if item_idx < len(relevance) else 0
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    
    return dcg


def calculate_ndcg(ranking: List[int], relevance: List[float], k: int = 5) -> float:
    """Calculate Normalized DCG."""
    dcg = calculate_dcg(ranking, relevance, k)
    
    # Ideal DCG (sorted relevance)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = calculate_dcg(list(range(1, len(ideal_relevance) + 1)), ideal_relevance, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def detect_language(text: str) -> str:
    """
    Simple language detection based on common words.
    Returns ISO 639-1 code.
    """
    text_lower = text.lower()
    
    # Portuguese indicators
    pt_words = ['que', 'não', 'uma', 'para', 'com', 'está', 'são', 'foi', 'muito']
    pt_count = sum(1 for w in pt_words if f' {w} ' in f' {text_lower} ')
    
    # English indicators  
    en_words = ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'been', 'with']
    en_count = sum(1 for w in en_words if f' {w} ' in f' {text_lower} ')
    
    if pt_count > en_count:
        return "pt"
    return "en"


def format_captions_for_prompt(captions: List[str], image_names: Optional[List[str]] = None) -> str:
    """Format image captions for inclusion in prompts."""
    lines = []
    for i, caption in enumerate(captions, 1):
        name = image_names[i-1] if image_names else f"Image {i}"
        lines.append(f"Image {i} ({name}): {caption}")
    return "\n".join(lines)


def safe_api_call(func, *args, max_retries: int = 3, **kwargs):
    """Wrapper for API calls with retry logic."""
    import time
    
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    raise last_error


