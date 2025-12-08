"""
AdMIRe 2.0 Data Loader
======================
Handles loading and processing of AdMIRe 2.0 datasets.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import os


@dataclass
class AdMIReItem:
    """Single item from the AdMIRe dataset."""
    # Core fields
    compound: str
    sentence: str
    
    # Image information (5 images)
    image_names: List[str]
    image_captions: List[str]
    image_paths: List[str]  # Full paths to images
    
    # Optional fields (may not be present in test data)
    sentence_type: Optional[str] = None  # "idiomatic" or "literal"
    expected_order: Optional[List[str]] = None  # Gold ranking
    
    # Metadata
    subset: str = "train"
    language: str = "en"
    item_id: Optional[str] = None
    
    # Portuguese-specific captions (if available)
    image_captions_pt: Optional[List[str]] = None
    
    def get_captions(self, prefer_language: str = "en") -> List[str]:
        """Get captions in preferred language."""
        if prefer_language == "pt" and self.image_captions_pt:
            return self.image_captions_pt
        return self.image_captions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compound": self.compound,
            "sentence": self.sentence,
            "image_names": self.image_names,
            "image_captions": self.image_captions,
            "sentence_type": self.sentence_type,
            "expected_order": self.expected_order,
            "language": self.language
        }


@dataclass  
class SubtaskBItem:
    """Single item from Subtask B (sequence completion)."""
    compound: str
    
    # Sequence images (2 images that form a sequence)
    sequence_captions: List[str]  # [caption1, caption2]
    sequence_paths: List[str]  # Paths to s1.png, s2.png
    
    # Candidate images (4 options)
    candidate_names: List[str]
    candidate_captions: List[str]
    candidate_paths: List[str]
    
    # Labels (not in test)
    sentence_type: Optional[str] = None
    expected_item: Optional[str] = None  # Correct candidate filename
    
    subset: str = "train"
    language: str = "en"


class AdMIReDataLoader:
    """
    Data loader for AdMIRe 2.0 datasets.
    
    Handles both Subtask A (image ranking) and Subtask B (sequence completion).
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
    
    def load_subtask_a(
        self, 
        language: str = "english",
        split: str = "train"
    ) -> List[AdMIReItem]:
        """
        Load Subtask A data for a specific language and split.
        
        Args:
            language: "english" or "portuguese"
            split: "train", "dev", or "test"
        
        Returns:
            List of AdMIReItem objects
        """
        # Determine file path
        lang_path = self.base_path / "subtask_a" / language
        tsv_file = lang_path / f"subtask_a_{split}.tsv"
        
        if not tsv_file.exists():
            print(f"Warning: {tsv_file} not found. Returning empty list.")
            return []
        
        # Read TSV
        df = pd.read_csv(tsv_file, sep='\t')
        
        items = []
        for _, row in df.iterrows():
            # Extract image names and captions
            image_names = [row[f"image{i}_name"] for i in range(1, 6)]
            image_captions = [row[f"image{i}_caption"] for i in range(1, 6)]
            
            # Build full image paths
            compound_folder = lang_path / row["compound"]
            image_paths = [str(compound_folder / name) for name in image_names]
            
            # Portuguese captions if available
            image_captions_pt = None
            if f"image1_caption_pt" in df.columns:
                image_captions_pt = [row[f"image{i}_caption_pt"] for i in range(1, 6)]
            
            # Parse expected order if present
            expected_order = None
            if "expected_order" in df.columns and pd.notna(row.get("expected_order")):
                order_str = str(row["expected_order"])
                # Handle various formats: "img1,img2,..." or "[img1, img2,...]"
                order_str = order_str.strip("[]")
                expected_order = [s.strip().strip("'\"") for s in order_str.split(",")]
            
            item = AdMIReItem(
                compound=row["compound"],
                sentence=row["sentence"],
                image_names=image_names,
                image_captions=image_captions,
                image_paths=image_paths,
                sentence_type=row.get("sentence_type"),
                expected_order=expected_order,
                subset=row.get("subset", split),
                language="pt" if language == "portuguese" else "en",
                item_id=f"{language}_{row.name}",
                image_captions_pt=image_captions_pt
            )
            items.append(item)
        
        return items
    
    def load_subtask_b(self, split: str = "train") -> List[SubtaskBItem]:
        """
        Load Subtask B data (sequence completion).
        
        Args:
            split: "train", "dev", or "test"
        
        Returns:
            List of SubtaskBItem objects
        """
        tsv_file = self.base_path / "subtask_b" / f"subtask_b_{split}.tsv"
        
        if not tsv_file.exists():
            print(f"Warning: {tsv_file} not found. Returning empty list.")
            return []
        
        df = pd.read_csv(tsv_file, sep='\t')
        
        items = []
        for _, row in df.iterrows():
            compound_folder = self.base_path / "subtask_b" / row["compound"]
            
            # Sequence images
            sequence_captions = [
                row["sequence_caption1"],
                row["sequence_caption2"]
            ]
            sequence_paths = [
                str(compound_folder / "s1.png"),
                str(compound_folder / "s2.png")
            ]
            
            # Candidate images (4 options)
            candidate_names = [row[f"image{i}_name"] for i in range(1, 5)]
            candidate_captions = [row[f"image{i}_caption"] for i in range(1, 5)]
            candidate_paths = [str(compound_folder / name) for name in candidate_names]
            
            item = SubtaskBItem(
                compound=row["compound"],
                sequence_captions=sequence_captions,
                sequence_paths=sequence_paths,
                candidate_names=candidate_names,
                candidate_captions=candidate_captions,
                candidate_paths=candidate_paths,
                sentence_type=row.get("sentence_type"),
                expected_item=row.get("expected_item"),
                subset=row.get("subset", split)
            )
            items.append(item)
        
        return items
    
    def iterate_all_languages(
        self, 
        split: str = "train"
    ) -> Iterator[AdMIReItem]:
        """Iterate over all available languages for Subtask A."""
        subtask_a_path = self.base_path / "subtask_a"
        
        if not subtask_a_path.exists():
            return
        
        for lang_folder in subtask_a_path.iterdir():
            if lang_folder.is_dir():
                items = self.load_subtask_a(lang_folder.name, split)
                yield from items
    
    def get_available_languages(self) -> List[str]:
        """Get list of available languages in the dataset."""
        subtask_a_path = self.base_path / "subtask_a"
        
        if not subtask_a_path.exists():
            return []
        
        return [f.name for f in subtask_a_path.iterdir() if f.is_dir()]


def create_mock_data() -> List[AdMIReItem]:
    """
    Create mock data for testing when real data is not available.
    
    Returns sample items that match the AdMIRe 2.0 format.
    """
    mock_items = [
        AdMIReItem(
            compound="green fingers",
            sentence="My grandmother has green fingers; her garden is always beautiful.",
            image_names=["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"],
            image_captions=[
                "A person gardening with plants and flowers around them",
                "Hands covered in green paint",
                "A beautiful garden with various flowers",
                "Green colored human fingers",
                "Someone planting seeds in soil"
            ],
            image_paths=["mock/img1.png"] * 5,
            sentence_type="idiomatic",
            expected_order=["img1.png", "img3.png", "img5.png", "img2.png", "img4.png"],
            language="en",
            item_id="mock_1"
        ),
        AdMIReItem(
            compound="couch potato",
            sentence="After work, he becomes a couch potato, watching TV for hours.",
            image_names=["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"],
            image_captions=[
                "A potato sitting on a couch",
                "A lazy person lying on a sofa watching television",
                "Someone exercising in a gym",
                "A person relaxing on a couch with snacks",
                "Potatoes in a garden"
            ],
            image_paths=["mock/img1.png"] * 5,
            sentence_type="idiomatic",
            expected_order=["img2.png", "img4.png", "img1.png", "img5.png", "img3.png"],
            language="en",
            item_id="mock_2"
        ),
        AdMIReItem(
            compound="green fingers",
            sentence="The child dipped their green fingers into the paint bucket.",
            image_names=["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"],
            image_captions=[
                "A person gardening with plants and flowers around them",
                "Hands covered in green paint",
                "A beautiful garden with various flowers",
                "Green colored human fingers",
                "Someone planting seeds in soil"
            ],
            image_paths=["mock/img1.png"] * 5,
            sentence_type="literal",
            expected_order=["img2.png", "img4.png", "img1.png", "img3.png", "img5.png"],
            language="en",
            item_id="mock_3"
        ),
    ]
    
    return mock_items


def create_mock_subtask_b_data() -> List[SubtaskBItem]:
    """Create mock data for Subtask B testing."""
    return [
        SubtaskBItem(
            compound="guinea pig",
            sequence_captions=[
                "A scientist in a laboratory preparing an experiment",
                "A person volunteering for a medical trial"
            ],
            sequence_paths=["mock/s1.png", "mock/s2.png"],
            candidate_names=["c1.png", "c2.png", "c3.png", "c4.png"],
            candidate_captions=[
                "A small furry rodent in a cage",
                "A person receiving an injection",
                "Scientists analyzing test results",
                "A pet store with various animals"
            ],
            candidate_paths=["mock/c1.png"] * 4,
            sentence_type="idiomatic",
            expected_item="c2.png"
        )
    ]

