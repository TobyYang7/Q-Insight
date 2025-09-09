#!/usr/bin/env python3
"""
Create comparison dataset for image quality comparison task.

This script:
1. Scans the dataset directory structure
2. Splits the data into 80% train and 20% test
3. Creates comparison pairs with quality ordering: weak < base < good
4. Generates JSON files for training and testing
"""

import os
import json
import random
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Configuration
DATASET_ROOT = "/Q-Insight/dataset/compare"
CATEGORIES = ["weak", "base", "good"]  # Quality order: weak < base < good
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
OUTPUT_DIR = "/Q-Insight/dataset/compare"

# Quality mapping for comparison results
QUALITY_ORDER = {"weak": 0, "base": 1, "good": 2}

def scan_dataset_structure() -> Dict[str, Dict[str, List[str]]]:
    """
    Scan the dataset directory and return the structure.
    
    Returns:
        Dict with structure: {category: {id_folder: [image_files]}}
    """
    dataset_structure = {}
    
    for category in CATEGORIES:
        category_path = os.path.join(DATASET_ROOT, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} not found at {category_path}")
            continue
            
        dataset_structure[category] = {}
        
        # Find all id folders in this category
        id_folders = glob.glob(os.path.join(category_path, "id*"))
        
        for id_folder in id_folders:
            folder_name = os.path.basename(id_folder)
            
            # Find all image files in this id folder
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg']:
                image_files.extend(glob.glob(os.path.join(id_folder, f"*{ext}")))
            
            # Convert to relative paths
            relative_files = []
            for img_file in image_files:
                rel_path = os.path.relpath(img_file, DATASET_ROOT)
                relative_files.append(rel_path)
            
            if relative_files:
                dataset_structure[category][folder_name] = sorted(relative_files)
    
    return dataset_structure

def get_common_ids(dataset_structure: Dict[str, Dict[str, List[str]]]) -> Set[str]:
    """
    Get IDs that exist in all categories.
    
    Args:
        dataset_structure: The scanned dataset structure
        
    Returns:
        Set of common ID folder names
    """
    category_ids = []
    for category, id_dict in dataset_structure.items():
        category_ids.append(set(id_dict.keys()))
    
    # Find intersection of all categories
    common_ids = set.intersection(*category_ids) if category_ids else set()
    return common_ids

def split_ids_train_test(common_ids: Set[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Split IDs into train and test sets.
    
    Args:
        common_ids: Set of common ID folder names
        train_ratio: Ratio for training set
        
    Returns:
        Tuple of (train_ids, test_ids)
    """
    ids_list = sorted(list(common_ids))
    random.shuffle(ids_list)
    
    train_size = int(len(ids_list) * train_ratio)
    train_ids = ids_list[:train_size]
    test_ids = ids_list[train_size:]
    
    return train_ids, test_ids

def create_comparison_pairs(dataset_structure: Dict[str, Dict[str, List[str]]], 
                          ids: List[str]) -> List[Dict[str, str]]:
    """
    Create comparison pairs for the given IDs.
    
    Args:
        dataset_structure: The scanned dataset structure
        ids: List of ID folder names to process
        
    Returns:
        List of comparison samples
    """
    comparison_samples = []
    
    for id_folder in ids:
        # Get all images for this ID across categories
        id_images = {}
        for category in CATEGORIES:
            if id_folder in dataset_structure[category]:
                id_images[category] = dataset_structure[category][id_folder]
        
        # Create comparison pairs
        for category in CATEGORIES:
            if category not in id_images:
                continue
                
            for img_file in id_images[category]:
                # Create multiple comparison pairs for each image
                pairs_created = 0
                max_pairs_per_image = 3  # Limit pairs per image to avoid too many samples
                
                while pairs_created < max_pairs_per_image:
                    # Randomly select two different categories for comparison
                    available_categories = [c for c in CATEGORIES if c != category and c in id_images]
                    if len(available_categories) < 2:
                        break
                    
                    # Select two categories randomly
                    cat_a, cat_b = random.sample(available_categories, 2)
                    
                    # Find corresponding images (same slide number)
                    img_name = os.path.basename(img_file)
                    img_a = None
                    img_b = None
                    
                    for img in id_images[cat_a]:
                        if os.path.basename(img) == img_name:
                            img_a = img
                            break
                    
                    for img in id_images[cat_b]:
                        if os.path.basename(img) == img_name:
                            img_b = img
                            break
                    
                    if img_a and img_b:
                        # Determine the result based on quality order
                        quality_a = QUALITY_ORDER[cat_a]
                        quality_b = QUALITY_ORDER[cat_b]
                        
                        if quality_a > quality_b:
                            result = "Slide A"
                        elif quality_b > quality_a:
                            result = "Slide B"
                        else:
                            result = "Similar"
                        
                        sample = {
                            "ref_image": img_file,  # Original image as reference
                            "ImageA": img_a,
                            "ImageB": img_b,
                            "result": result
                        }
                        
                        comparison_samples.append(sample)
                        pairs_created += 1
                    else:
                        break
    
    return comparison_samples

def save_dataset(samples: List[Dict[str, str]], output_path: str):
    """
    Save the dataset to a JSON file.
    
    Args:
        samples: List of comparison samples
        output_path: Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to {output_path}")

def print_dataset_statistics(dataset_structure: Dict[str, Dict[str, List[str]]], 
                           train_ids: List[str], test_ids: List[str],
                           train_samples: List[Dict[str, str]], 
                           test_samples: List[Dict[str, str]]):
    """
    Print statistics about the created dataset.
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nCategories: {CATEGORIES}")
    print(f"Quality order: {' < '.join(CATEGORIES)}")
    
    print(f"\nTotal IDs found: {len(train_ids) + len(test_ids)}")
    print(f"Train IDs: {len(train_ids)}")
    print(f"Test IDs: {len(test_ids)}")
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Count results distribution
    train_results = defaultdict(int)
    test_results = defaultdict(int)
    
    for sample in train_samples:
        train_results[sample['result']] += 1
    
    for sample in test_samples:
        test_results[sample['result']] += 1
    
    print(f"\nTrain result distribution:")
    for result, count in train_results.items():
        print(f"  {result}: {count}")
    
    print(f"\nTest result distribution:")
    for result, count in test_results.items():
        print(f"  {result}: {count}")

def main():
    """Main function to create the comparison dataset."""
    print("Creating comparison dataset...")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Scan dataset structure
    print("\nStep 1: Scanning dataset structure...")
    dataset_structure = scan_dataset_structure()
    
    if not dataset_structure:
        print("Error: No dataset structure found!")
        return
    
    # Print structure info
    for category, id_dict in dataset_structure.items():
        print(f"  {category}: {len(id_dict)} IDs")
    
    # Step 2: Get common IDs
    print("\nStep 2: Finding common IDs across categories...")
    common_ids = get_common_ids(dataset_structure)
    print(f"  Found {len(common_ids)} common IDs")
    
    if not common_ids:
        print("Error: No common IDs found across all categories!")
        return
    
    # Step 3: Split into train/test
    print("\nStep 3: Splitting into train/test sets...")
    train_ids, test_ids = split_ids_train_test(common_ids, TRAIN_RATIO)
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Test IDs: {len(test_ids)}")
    
    # Step 4: Create comparison pairs
    print("\nStep 4: Creating comparison pairs...")
    train_samples = create_comparison_pairs(dataset_structure, train_ids)
    test_samples = create_comparison_pairs(dataset_structure, test_ids)
    
    # Step 5: Save datasets
    print("\nStep 5: Saving datasets...")
    train_output = os.path.join(OUTPUT_DIR, "train_comparison.json")
    test_output = os.path.join(OUTPUT_DIR, "test_comparison.json")
    
    save_dataset(train_samples, train_output)
    save_dataset(test_samples, test_output)
    
    # Step 6: Print statistics
    print_dataset_statistics(dataset_structure, train_ids, test_ids, train_samples, test_samples)
    
    print(f"\nDataset creation completed!")
    print(f"Train dataset: {train_output}")
    print(f"Test dataset: {test_output}")

if __name__ == "__main__":
    main()
