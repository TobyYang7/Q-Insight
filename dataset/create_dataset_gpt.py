#!/usr/bin/env python3
"""
Create dataset using GPT API for image deficiency detection and scoring.

This script:
1. Scans all images in the compare folder
2. Calls GPT API to analyze each image for deficiencies and scoring
3. Saves results in the same format as deficiency_train.json and score_train.json
"""

import os
import json
import base64
import glob
import time
import random
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import openai
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configuration
DATASET_ROOT = "/root/Q-Insight/dataset/compare"
OUTPUT_DIR = "/root/Q-Insight/dataset"
CATEGORIES = ["weak", "base", "good", "gt_60"]
DEFAULT_WORKERS = 30
MAX_IMAGE_SIZE = 960  # Maximum pixel dimension


def get_openai_client():
    """Get OpenAI client instance."""
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE_URL")
    )


def resize_image(image_path: str, max_size: int = MAX_IMAGE_SIZE) -> bytes:
    """
    Resize image to fit within max_size while maintaining aspect ratio.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) in pixels
        
    Returns:
        Resized image as bytes
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Get current dimensions
            width, height = img.size

            # Calculate if resizing is needed
            if max(width, height) > max_size:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)

                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            return img_byte_arr.getvalue()

    except Exception as e:
        print(f"Error resizing image {image_path}: {str(e)}")
        # Fallback: read original file
        with open(image_path, "rb") as f:
            return f.read()


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string with size optimization.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the resized image
    """
    try:
        # Resize image first
        image_bytes = resize_image(image_path, MAX_IMAGE_SIZE)

        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')

    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        # Fallback: try to read original file
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as fallback_e:
            print(f"Fallback encoding also failed for {image_path}: {str(fallback_e)}")
            raise fallback_e


def get_deficiency_analysis(image_path: str, client: OpenAI = None) -> Dict[str, Any]:
    """
    Call GPT API to analyze image for deficiencies.
    
    Args:
        image_path: Path to the image file
        client: OpenAI client instance
        
    Returns:
        Dictionary containing deficiency analysis
    """
    try:
        if client is None:
            client = get_openai_client()

        # Encode image
        base64_image = encode_image(image_path)

        # Prepare the prompt
        prompt = """
        Analyze this presentation slide image for design deficiencies. 
        Look for issues in the following three categories:
        
        1. Composition & Layout:
           - Poor Visual Hierarchy
           - Content Alignment Issues
           - Content Overflow/Cut-off
           - Unbalanced Space Distribution
        
        2. Typography:
           - Illegible Typeface Selection or Usage
           - Improper Font Sizing
           - Excessive Text Volume
           - Improper Line/Character Spacing
        
        3. Imagery & Visualizations:
           - Irrelevant Visual Content
           - Improper Image Sizing
           - Inconsistent Visual Style Usage
           - Inappropriate or Mismatched Color Combinations
        
        Return your analysis in the following JSON format:
        {
            "deficiency_count": <number of deficiencies found>,
            "deficiencies": [
                {
                    "category": "<exact category name from above>",
                    "deficiency": "<specific deficiency description matching the types listed>",
                    "has_strong_agreement": <true if this is clearly a deficiency, false if subjective>
                }
            ]
        }
        
        If no deficiencies are found, return deficiency_count: 0 and empty deficiencies array.
        Be thorough but fair in your assessment. Use the exact category names and deficiency types listed above.
        """

        response = client.chat.completions.create(
            model=os.getenv("MODEL_TYPE", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1000,
            temperature=0.9
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)

        # Ensure all required keys are present
        if not all(key in result for key in ["deficiency_count", "deficiencies"]):
            print(f"Warning: Incomplete deficiency response for {image_path}, using defaults")
            return {
                "deficiency_count": 0,
                "deficiencies": []
            }

        return result

    except Exception as e:
        print(f"Error analyzing {image_path}: {str(e)}")
        return {
            "deficiency_count": 0,
            "deficiencies": []
        }


def get_single_score_analysis(image_path: str, client: OpenAI = None) -> Dict[str, Any]:
    """
    Call GPT API to score the image once.
    
    Args:
        image_path: Path to the image file
        client: OpenAI client instance
        
    Returns:
        Dictionary containing score analysis
    """
    try:
        if client is None:
            client = get_openai_client()

        # Encode image
        base64_image = encode_image(image_path)

        # Shortened prompt to reduce token usage
        prompt = """
        Rate this slide (1-5 scale, 2 decimal places) using STRICT criteria:

        1. Composition & Layout: hierarchy, alignment, spacing
        2. Typography: readability, sizing, spacing  
        3. Imagery: relevance, sizing, style consistency

        JSON format:
        {
            "composition": <score>,
            "typography": <score>, 
            "imagery": <score>
        }

        Strict scoring: 1=Poor, 2=Below avg, 3=Average, 4=Good, 5=Excellent
        Most slides should score 2-3 range. Perfect scores (5.00) rare.
        Any major flaw caps score at 3 or below.
        """

        response = client.chat.completions.create(
            model=os.getenv("MODEL_TYPE", "gpt-4o"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.9
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content)

        # Ensure all required keys are present
        if not all(key in result for key in ["composition", "typography", "imagery"]):
            print(f"Warning: Incomplete score response for {image_path}, using defaults")
            return {
                "composition": 3.0,
                "typography": 3.0,
                "imagery": 3.0
            }

        return result

    except Exception as e:
        print(f"Error scoring {image_path}: {str(e)}")
        return {
            "composition": 3.0,
            "typography": 3.0,
            "imagery": 3.0
        }


def get_score_analysis(image_path: str, client: OpenAI = None) -> Dict[str, Any]:
    """
    Call GPT API to score the image 3 times and calculate average for each dimension.
    Each dimension is scored separately to avoid interference.
    
    Args:
        image_path: Path to the image file
        client: OpenAI client instance
        
    Returns:
        Dictionary containing averaged score analysis
    """
    if client is None:
        client = get_openai_client()

    # Collect scores from 3 separate evaluations
    all_scores = {
        "composition": [],
        "typography": [],
        "imagery": []
    }

    # Get 3 complete evaluations
    for i in range(3):
        score = get_single_score_analysis(image_path, client)
        all_scores["composition"].append(score.get("composition", 3.0))
        all_scores["typography"].append(score.get("typography", 3.0))
        all_scores["imagery"].append(score.get("imagery", 3.0))

        # Delay between complete evaluations (not between dimensions within same evaluation)
        if i < 2:  # Don't delay after the last iteration
            time.sleep(0.3)

    # Calculate averages
    avg_composition = sum(all_scores["composition"]) / 3
    avg_typography = sum(all_scores["typography"]) / 3
    avg_imagery = sum(all_scores["imagery"]) / 3
    overall = (avg_composition + avg_typography + avg_imagery) / 3

    return {
        "composition": round(avg_composition, 2),
        "typography": round(avg_typography, 2),
        "imagery": round(avg_imagery, 2),
        "overall": round(overall, 2)
    }


def scan_all_images() -> List[str]:
    """
    Scan all images in the compare folder.
    
    Returns:
        List of image file paths
    """
    image_files = []

    for category in CATEGORIES:
        category_path = os.path.join(DATASET_ROOT, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} not found at {category_path}")
            continue

        # Find all image files in this category
        for ext in ['.png', '.jpg', '.jpeg']:
            pattern = os.path.join(category_path, "**", f"*{ext}")
            files = glob.glob(pattern, recursive=True)
            image_files.extend(files)

    return sorted(image_files)


def process_single_image(args: Tuple[str, int, int]) -> Tuple[Dict, Dict]:
    """
    Process a single image for both deficiency and score analysis.
    
    Args:
        args: Tuple of (image_path, image_index, total_images)
        
    Returns:
        Tuple of (deficiency_result, score_result)
    """
    image_path, image_index, total_images = args

    # Get relative path for saving
    rel_path = os.path.relpath(image_path, "/root/Q-Insight")

    # Extract slide_id from filename
    slide_id = os.path.splitext(os.path.basename(image_path))[0]

    # Create client for this process
    client = get_openai_client()

    # Get score analysis first
    score_analysis = get_score_analysis(image_path, client)
    score_result = {
        "image": rel_path,
        "score": score_analysis
    }

    # Only analyze deficiencies if any category score is below 3
    deficiency_result = {
        "image": rel_path,
        "slide_id": slide_id,
        "deficiency_count": 0,
        "deficiencies": []
    }

    # Check if any category score is below 3
    if (score_analysis.get("composition", 3) < 3 or
        score_analysis.get("typography", 3) < 3 or
            score_analysis.get("imagery", 3) < 3):

        deficiency_analysis = get_deficiency_analysis(image_path, client)
        deficiency_result = {
            "image": rel_path,
            "slide_id": slide_id,
            "deficiency_count": deficiency_analysis.get("deficiency_count", 0),
            "deficiencies": deficiency_analysis.get("deficiencies", [])
        }

    return deficiency_result, score_result


def process_images(image_files: List[str], workers: int = DEFAULT_WORKERS) -> tuple:
    """
    Process images with GPT API calls using multiprocessing.
    
    Args:
        image_files: List of image file paths
        workers: Number of worker processes
        
    Returns:
        Tuple of (deficiency_results, score_results)
    """
    total_files = len(image_files)
    print(f"Processing {total_files} images with {workers} workers...")

    deficiency_results = []
    score_results = []

    # Create main progress bar for individual images
    with tqdm(total=total_files, desc="Processing images", unit="image") as pbar:
        # Use multiprocessing Pool
        with mp.Pool(processes=workers) as pool:
            # Prepare arguments for all images
            all_args = [(image_path, i, total_files) for i, image_path in enumerate(image_files)]

            # Use imap to get results as they complete
            for deficiency_result, score_result in pool.imap(process_single_image, all_args):
                deficiency_results.append(deficiency_result)
                score_results.append(score_result)

                # Update progress bar for each completed image
                pbar.update(1)
                pbar.set_postfix_str(f"Completed: {len(deficiency_results)}/{total_files}")

    return deficiency_results, score_results


def save_results(deficiency_results: List[Dict], score_results: List[Dict], processed_count: int = None):
    """
    Save results to JSON files.
    
    Args:
        deficiency_results: List of deficiency analysis results
        score_results: List of score analysis results
        processed_count: Number of processed images (for filename)
    """
    suffix = f"_{processed_count}" if processed_count else ""

    # Save deficiency results
    deficiency_file = os.path.join(OUTPUT_DIR, f"deficiency_gpt{suffix}.json")
    with open(deficiency_file, 'w', encoding='utf-8') as f:
        json.dump(deficiency_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(deficiency_results)} deficiency results to {deficiency_file}")

    # Save score results
    score_file = os.path.join(OUTPUT_DIR, f"score_gpt{suffix}.json")
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(score_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(score_results)} score results to {score_file}")


def save_final_results(deficiency_results: List[Dict], score_results: List[Dict]):
    """
    Save final results to the standard filenames.
    
    Args:
        deficiency_results: List of deficiency analysis results
        score_results: List of score analysis results
    """
    # Save deficiency results
    deficiency_file = os.path.join(OUTPUT_DIR, "deficiency_gpt.json")
    with open(deficiency_file, 'w', encoding='utf-8') as f:
        json.dump(deficiency_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(deficiency_results)} deficiency results to {deficiency_file}")

    # Save score results
    score_file = os.path.join(OUTPUT_DIR, "score_gpt.json")
    with open(score_file, 'w', encoding='utf-8') as f:
        json.dump(score_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(score_results)} score results to {score_file}")


def main(workers: int = DEFAULT_WORKERS):
    """Main function to process images with GPT API."""
    print("Starting GPT-based dataset creation...")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Workers: {workers}")
    print(f"Max image size: {MAX_IMAGE_SIZE} pixels")

    # Check if required environment variables are available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables!")
        return

    if not os.getenv("OPENAI_API_BASE_URL"):
        print("Error: OPENAI_API_BASE_URL not found in environment variables!")
        return

    if not os.getenv("MODEL_TYPE"):
        print("Error: MODEL_TYPE not found in environment variables!")
        return

    # Scan all images
    print("\nScanning for images...")
    image_files = scan_all_images()
    print(f"Found {len(image_files)} images to process")

    if not image_files:
        print("No images found!")
        return

    # Process images
    print("\nProcessing images with GPT API...")
    deficiency_results, score_results = process_images(image_files, workers)

    # Save final results
    print("\nSaving final results...")
    save_final_results(deficiency_results, score_results)

    print(f"\nDataset creation completed!")
    print(f"Processed {len(image_files)} images")
    print(f"Deficiency results: {len(deficiency_results)} entries")
    print(f"Score results: {len(score_results)} entries")
    print(f"Output files:")
    print(f"  - deficiency_gpt.json")
    print(f"  - score_gpt.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset using GPT API for image analysis")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of worker processes (default: {DEFAULT_WORKERS})")

    args = parser.parse_args()
    main(workers=args.workers)
