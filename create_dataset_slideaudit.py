#!/usr/bin/env python3
"""
SlideAudit Dataset Creator
Creates simplified train/test dataset JSON files from SlideAudit annotations.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import random

def load_annotations(annotations_dir: str) -> Dict[str, Dict]:
    """Load all annotation files and return a dictionary mapping slide_id to annotation data."""
    annotations = {}
    
    for json_file in glob.glob(os.path.join(annotations_dir, "*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                slide_id = data.get('slide_id')
                if slide_id:
                    annotations[str(slide_id)] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return annotations

def create_slide_entry(slide_id: str, annotation_data: Dict) -> Dict[str, Any]:
    """Create a slide entry with design deficiency information."""
    
    # Extract actual deficiencies (where response is true)
    deficiencies = []
    for annotation in annotation_data.get('annotations', []):
        if annotation.get('response', False):  # Only include actual deficiencies
            deficiency_info = {
                'category': annotation.get('design_deficiency_category'),
                'deficiency': annotation.get('design_deficiency'),
                'has_strong_agreement': annotation.get('has_strong_agreement', False),
                'bounding_boxes': annotation.get('bounding_boxes', [])
            }
            deficiencies.append(deficiency_info)
    
    # Create slide entry with actual annotation data
    slide_entry = {
        "image": f"SlideAudit/data/images/slide_{slide_id.zfill(4)}.jpg",
        "slide_id": slide_id,
        "has_design_deficiencies": len(deficiencies) > 0,
        "deficiency_count": len(deficiencies),
        "deficiencies": deficiencies
    }
    
    return slide_entry

def create_train_test_datasets(annotations_dir: str, train_ratio: float = 0.8):
    """Create train and test datasets with the specified split ratio."""
    
    print("Loading SlideAudit annotations...")
    annotations = load_annotations(annotations_dir)
    print(f"Loaded {len(annotations)} annotation files")
    
    print("Creating dataset entries...")
    dataset = []
    
    for slide_id in sorted(annotations.keys(), key=lambda x: int(x)):
        annotation_data = annotations[slide_id]
        slide_entry = create_slide_entry(slide_id, annotation_data)
        dataset.append(slide_entry)
    
    # Shuffle dataset for random split
    random.seed(42)  # For reproducible results
    random.shuffle(dataset)
    
    # Split into train and test
    split_index = int(len(dataset) * train_ratio)
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    
    print(f"Dataset split: {len(train_data)} train, {len(test_data)} test")
    
    # Save train dataset
    train_file = "slideaudit_train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Train dataset saved to: {train_file}")
    
    # Save test dataset
    test_file = "slideaudit_test.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"Test dataset saved to: {test_file}")
    
    # Print data distribution
    print_data_distribution(train_data, test_data)
    
    return train_data, test_data

def print_data_distribution(train_data: List[Dict], test_data: List[Dict]):
    """Print the distribution of data across deficiency counts for both train and test sets."""
    
    print("\n=== Data Distribution ===")
    
    # Train set distribution by deficiency count
    train_dist = defaultdict(int)
    for item in train_data:
        deficiency_count = item['deficiency_count']
        train_dist[deficiency_count] += 1
    
    print(f"\nTrain Set ({len(train_data)} samples):")
    for deficiency_count in sorted(train_dist.keys()):
        count = train_dist[deficiency_count]
        percentage = (count / len(train_data)) * 100
        print(f"  {deficiency_count} deficiencies: {count} samples ({percentage:.1f}%)")
    
    # Test set distribution by deficiency count
    test_dist = defaultdict(int)
    for item in test_data:
        deficiency_count = item['deficiency_count']
        test_dist[deficiency_count] += 1
    
    print(f"\nTest Set ({len(test_data)} samples):")
    for deficiency_count in sorted(test_dist.keys()):
        count = test_dist[deficiency_count]
        percentage = (count / len(test_data)) * 100
        print(f"  {deficiency_count} deficiencies: {count} samples ({percentage:.1f}%)")
    
    # Overall distribution
    all_data = train_data + test_data
    overall_dist = defaultdict(int)
    for item in all_data:
        deficiency_count = item['deficiency_count']
        overall_dist[deficiency_count] += 1
    
    print(f"\nOverall Dataset ({len(all_data)} samples):")
    for deficiency_count in sorted(overall_dist.keys()):
        count = overall_dist[deficiency_count]
        percentage = (count / len(all_data)) * 100
        print(f"  {deficiency_count} deficiencies: {count} samples ({percentage:.1f}%)")
    
    # Deficiency category distribution
    print(f"\n=== Deficiency Category Distribution ===")
    category_stats = defaultdict(int)
    for item in all_data:
        for deficiency in item['deficiencies']:
            category = deficiency['category']
            category_stats[category] += 1
    
    for category in sorted(category_stats.keys()):
        count = category_stats[category]
        print(f"  {category}: {count} deficiencies")

def main():
    """Main function to create the SlideAudit train/test datasets."""
    
    # Define paths
    current_dir = Path(__file__).parent
    annotations_dir = current_dir / "SlideAudit" / "data" / "annotations"
    
    # Check if annotations directory exists
    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return
    
    # Create train/test datasets
    train_data, test_data = create_train_test_datasets(str(annotations_dir), train_ratio=0.8)
    
    print(f"\n=== Summary ===")
    print(f"Total slides processed: {len(train_data) + len(test_data)}")
    print(f"Train set: {len(train_data)} slides")
    print(f"Test set: {len(test_data)} slides")
    print(f"Train/Test ratio: {len(train_data)/(len(train_data) + len(test_data)):.1f}")

if __name__ == "__main__":
    main()

