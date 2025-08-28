import json
import os
import base64
import io
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
from PIL import Image
import openai
from pydantic import BaseModel
import time

# --- Deficiency Categories and Mapping ---
# Define categories and create a reverse map for easy lookup.
# This is defined globally so it doesn't need to be recreated in every call.
DEFICIENCY_CATEGORIES = {
    "Composition & Layout": [
        "Poor Visual Hierarchy",
        "Cluttered Layout",
        "Unbalanced Space Distribution",
        "Content Alignment Issues",
        "Content Overflow/Cut-off",
        "Occluded Content"
    ],
    "Typography": [
        "Illegible Typeface Selection or Usage",
        "Improper Font Sizing",
        "Excessive Text Volume",
        "Improper Text Styling",
        "Improper Line/Character Spacing",
        "Poor Text Hierarchy"
    ],
    "Color": [
        "Insufficient Color Contrast for Readability",
        "Excessive or Inconsistent Color Usage",
        "Inappropriate or Mismatched Color Combinations"
    ],
    "Imagery & Visualizations": [
        "Irrelevant Visual Content",
        "Poor Image Quality/Editing",
        "Improper Image Sizing",
        "Inconsistent Visual Style Usage"
    ]
}

DEFICIENCY_TO_CATEGORY_MAP = {
    deficiency: category
    for category, deficiencies in DEFICIENCY_CATEGORIES.items()
    for deficiency in deficiencies
}

# Get all unique deficiencies for metrics initialization
ALL_DEFICIENCIES = list(DEFICIENCY_TO_CATEGORY_MAP.keys())


# Load environment variables from .env file manually
def load_env_file(env_file='.env'):
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


load_env_file()


def extract_answer_content(text: str) -> str:
    """Extract content from <answer></answer> tags"""
    import re
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class DeficiencyClassification(BaseModel):
    deficiencies: List[str]


def image_to_base64_uri(image_path: str, max_width: int = 960) -> str:
    """
    Loads an image, resizes it to a maximum width while preserving
    the aspect ratio, and encodes it as a Base64 data URI.
    """
    try:
        img = Image.open(image_path)
        img_format = img.format if img.format else 'PNG'

        if img.width > max_width:
            aspect_ratio = img.height / img.width
            new_height = int(max_width * aspect_ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format=img_format)
        img_byte = buffered.getvalue()
        base64_str = base64.b64encode(img_byte).decode('utf-8')

        return f"data:image/{img_format.lower()};base64,{base64_str}"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def classify_deficiencies(model_output_text: str) -> List[str]:
    """
    Uses OpenAI API to classify deficiencies in model output text.
    Returns a list of classified deficiencies.
    """
    prompt = f"""Analyze the input text which describes slide design problems. From the predefined categories, identify all deficiencies mentioned in the text.

    Predefined deficiency categories:
    {json.dumps(ALL_DEFICIENCIES, indent=2)}

    Input text to analyze:
    {extract_answer_content(model_output_text)}

    Respond with a JSON object containing a single key "deficiencies" which holds a list of the exact names of the deficiencies found. If no deficiencies are found, the list should be empty."""

    try:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL")
        )

        model_type = os.getenv("MODEL_TYPE", "gpt-4o-2024-08-06")

        completion = client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "You are an expert in slide design analysis. Your output must be a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        response_text = completion.choices[0].message.content
        if response_text:
            parsed_response = DeficiencyClassification.model_validate_json(response_text)
            return parsed_response.deficiencies
        else:
            return []

    except Exception as e:
        print(f"Error calling OpenAI API or parsing response: {e}")
        return []


class DeficiencyTester:
    def __init__(self):
        # Load API configuration
        self.test_model = os.getenv("TEST_MODEL")
        self.test_api_key = os.getenv("TEST_API_KEY")
        self.test_base_url = os.getenv("TEST_BASE_URL")
        
        if not all([self.test_model, self.test_api_key, self.test_base_url]):
            raise ValueError("TEST_MODEL, TEST_API_KEY, and TEST_BASE_URL must be set in .env file")
        
        # Initialize OpenAI client for test model
        self.client = openai.OpenAI(
            api_key=self.test_api_key,
            base_url=self.test_base_url
        )

        self.system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )

        self.deficiency_prompt = (
            "What are the major design deficiencies in the slide? How do you think the slide can be improved to avoid these deficiencies? How can we adjust the elements on the slide to improve the slide?"
            "If there are no major deficiencies, simply respond with 'No deficiencies' without any other text."
        )

    def process_single(self, item: Dict, max_retries: int = 3) -> Dict:
        """Process a single test sample: call API, classify deficiencies, and calculate metrics."""
        try:
            # Convert image to base64
            base64_image_uri = image_to_base64_uri(item["image"])
            if not base64_image_uri:
                return None

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.deficiency_prompt},
                        {"type": "image_url", "image_url": {"url": base64_image_uri}}
                    ]
                }
            ]

            # Call API with retries
            generated_text = ""
            for attempt in range(max_retries):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.test_model,
                        messages=messages,
                        temperature=1.0,
                        top_p=0.95,
                        max_tokens=1024,
                    )
                    
                    generated_text = completion.choices[0].message.content or ""
                    break
                    
                except Exception as e:
                    print(f"API call failed for slide {item['slide_id']} (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"Failed after {max_retries} attempts for slide {item['slide_id']}")
                        return None
            
            # Classify specific deficiencies from the generated text
            predicted_deficiencies = classify_deficiencies(generated_text)

            # Get ground truth specific deficiencies
            ground_truth_deficiencies = [d["deficiency"] for d in item["deficiencies"]]

            # Convert specific deficiencies to major categories
            predicted_categories = {
                DEFICIENCY_TO_CATEGORY_MAP[d] for d in predicted_deficiencies 
                if d in DEFICIENCY_TO_CATEGORY_MAP
            }
            ground_truth_categories = {
                DEFICIENCY_TO_CATEGORY_MAP[d] for d in ground_truth_deficiencies 
                if d in DEFICIENCY_TO_CATEGORY_MAP
            }

            # Calculate metrics for CATEGORIES
            cat_true_positives = len(predicted_categories.intersection(ground_truth_categories))
            cat_false_positives = len(predicted_categories - ground_truth_categories)
            cat_false_negatives = len(ground_truth_categories - predicted_categories)

            cat_precision = cat_true_positives / (cat_true_positives + cat_false_positives) if (cat_true_positives + cat_false_positives) > 0 else 0
            cat_recall = cat_true_positives / (cat_true_positives + cat_false_negatives) if (cat_true_positives + cat_false_negatives) > 0 else 0
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0

            # Calculate metrics for SPECIFIC DEFICIENCIES
            predicted_deficiencies_set = set(predicted_deficiencies)
            ground_truth_deficiencies_set = set(ground_truth_deficiencies)
            
            def_true_positives = len(predicted_deficiencies_set.intersection(ground_truth_deficiencies_set))
            def_false_positives = len(predicted_deficiencies_set - ground_truth_deficiencies_set)
            def_false_negatives = len(ground_truth_deficiencies_set - predicted_deficiencies_set)

            def_precision = def_true_positives / (def_true_positives + def_false_positives) if (def_true_positives + def_false_positives) > 0 else 0
            def_recall = def_true_positives / (def_true_positives + def_false_negatives) if (def_true_positives + def_false_negatives) > 0 else 0
            def_f1 = 2 * def_precision * def_recall / (def_precision + def_recall) if (def_precision + def_recall) > 0 else 0

            result = {
                "slide_id": item["slide_id"],
                "image": item["image"],
                "ground_truth_deficiencies": ground_truth_deficiencies,
                "predicted_deficiencies": predicted_deficiencies,
                "ground_truth_categories": sorted(list(ground_truth_categories)),
                "predicted_categories": sorted(list(predicted_categories)),
                "model_output": generated_text,
                # Category-level metrics
                "category_metrics": {
                    "precision": cat_precision,
                    "recall": cat_recall,
                    "f1": cat_f1,
                    "true_positives": cat_true_positives,
                    "false_positives": cat_false_positives,
                    "false_negatives": cat_false_negatives
                },
                # Deficiency-level metrics
                "deficiency_metrics": {
                    "precision": def_precision,
                    "recall": def_recall,
                    "f1": def_f1,
                    "true_positives": def_true_positives,
                    "false_positives": def_false_positives,
                    "false_negatives": def_false_negatives
                }
            }

            print(f"Processed slide {item['slide_id']} - Cat F1: {cat_f1:.3f}, Def F1: {def_f1:.3f}")
            return result

        except Exception as e:
            print(f"Error processing slide {item['slide_id']}: {e}")
            return None


def calculate_per_class_metrics(results: List[Dict]) -> Dict:
    """Calculate per-category and per-deficiency metrics."""
    # Initialize counters for each category
    category_stats = {cat: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cat in DEFICIENCY_CATEGORIES.keys()}
    
    # Initialize counters for each specific deficiency
    deficiency_stats = {def_name: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for def_name in ALL_DEFICIENCIES}
    
    for result in results:
        # Process category-level stats
        gt_categories = set(result["ground_truth_categories"])
        pred_categories = set(result["predicted_categories"])
        
        for cat in DEFICIENCY_CATEGORIES.keys():
            if cat in gt_categories:
                category_stats[cat]["support"] += 1
                if cat in pred_categories:
                    category_stats[cat]["tp"] += 1
                else:
                    category_stats[cat]["fn"] += 1
            elif cat in pred_categories:
                category_stats[cat]["fp"] += 1
        
        # Process deficiency-level stats
        gt_deficiencies = set(result["ground_truth_deficiencies"])
        pred_deficiencies = set(result["predicted_deficiencies"])
        
        for def_name in ALL_DEFICIENCIES:
            if def_name in gt_deficiencies:
                deficiency_stats[def_name]["support"] += 1
                if def_name in pred_deficiencies:
                    deficiency_stats[def_name]["tp"] += 1
                else:
                    deficiency_stats[def_name]["fn"] += 1
            elif def_name in pred_deficiencies:
                deficiency_stats[def_name]["fp"] += 1
    
    # Calculate metrics for each category
    category_metrics = {}
    for cat, stats in category_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics[cat] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": stats["support"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    # Calculate metrics for each deficiency
    deficiency_metrics = {}
    for def_name, stats in deficiency_stats.items():
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        deficiency_metrics[def_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": stats["support"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    return {
        "category_metrics": category_metrics,
        "deficiency_metrics": deficiency_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Test deficiency detection with API")
    parser.add_argument("--test_data", type=str, default="slideaudit_test.json",
                        help="Path to test data JSON file")
    parser.add_argument("--output_file", type=str, default="deficiency_test_results_api.json",
                        help="Output file for results")
    parser.add_argument("--num_workers", type=int, default=30,
                        help="Number of concurrent API workers")

    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test samples")
    print(f"Using {args.num_workers} concurrent workers")
    print(f"Test Model: {os.getenv('TEST_MODEL')}")
    print(f"API Base URL: {os.getenv('TEST_BASE_URL')}")

    # Initialize tester
    tester = DeficiencyTester()
    
    all_results = []

    if args.num_workers == 1:
        # Single thread mode
        for item in tqdm(test_data, desc="Processing samples"):
            result = tester.process_single(item)
            if result:
                all_results.append(result)
    else:
        # Multi-threaded mode for concurrent API calls
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(tester.process_single, item): item for item in test_data}
            
            # Collect results with progress bar
            with tqdm(total=len(test_data), desc="Processing samples") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.append(result)
                    pbar.update(1)

    # Calculate overall metrics
    if all_results:
        # Category-level overall metrics
        cat_total_precision = sum(r["category_metrics"]["precision"] for r in all_results) / len(all_results)
        cat_total_recall = sum(r["category_metrics"]["recall"] for r in all_results) / len(all_results)
        cat_total_f1 = sum(r["category_metrics"]["f1"] for r in all_results) / len(all_results)

        cat_total_tp = sum(r["category_metrics"]["true_positives"] for r in all_results)
        cat_total_fp = sum(r["category_metrics"]["false_positives"] for r in all_results)
        cat_total_fn = sum(r["category_metrics"]["false_negatives"] for r in all_results)

        cat_overall_precision = cat_total_tp / (cat_total_tp + cat_total_fp) if (cat_total_tp + cat_total_fp) > 0 else 0
        cat_overall_recall = cat_total_tp / (cat_total_tp + cat_total_fn) if (cat_total_tp + cat_total_fn) > 0 else 0
        cat_overall_f1 = 2 * cat_overall_precision * cat_overall_recall / (cat_overall_precision + cat_overall_recall) if (cat_overall_precision + cat_overall_recall) > 0 else 0

        # Deficiency-level overall metrics
        def_total_precision = sum(r["deficiency_metrics"]["precision"] for r in all_results) / len(all_results)
        def_total_recall = sum(r["deficiency_metrics"]["recall"] for r in all_results) / len(all_results)
        def_total_f1 = sum(r["deficiency_metrics"]["f1"] for r in all_results) / len(all_results)

        def_total_tp = sum(r["deficiency_metrics"]["true_positives"] for r in all_results)
        def_total_fp = sum(r["deficiency_metrics"]["false_positives"] for r in all_results)
        def_total_fn = sum(r["deficiency_metrics"]["false_negatives"] for r in all_results)

        def_overall_precision = def_total_tp / (def_total_tp + def_total_fp) if (def_total_tp + def_total_fp) > 0 else 0
        def_overall_recall = def_total_tp / (def_total_tp + def_total_fn) if (def_total_tp + def_total_fn) > 0 else 0
        def_overall_f1 = 2 * def_overall_precision * def_overall_recall / (def_overall_precision + def_overall_recall) if (def_overall_precision + def_overall_recall) > 0 else 0
        
        # Calculate per-class metrics
        per_class_results = calculate_per_class_metrics(all_results)
        
    else:
        cat_total_precision = cat_total_recall = cat_total_f1 = 0
        cat_overall_precision = cat_overall_recall = cat_overall_f1 = 0
        cat_total_tp = cat_total_fp = cat_total_fn = 0
        
        def_total_precision = def_total_recall = def_total_f1 = 0
        def_overall_precision = def_overall_recall = def_overall_f1 = 0
        def_total_tp = def_total_fp = def_total_fn = 0
        
        per_class_results = {"category_metrics": {}, "deficiency_metrics": {}}

    # Prepare final results
    final_results = {
        "test_model": os.getenv("TEST_MODEL"),
        "test_api_base_url": os.getenv("TEST_BASE_URL"),
        "test_data": args.test_data,
        "total_samples": len(all_results),
        "overall_metrics": {
            "category_level": {
                "average_precision": cat_total_precision,
                "average_recall": cat_total_recall,
                "average_f1": cat_total_f1,
                "overall_precision": cat_overall_precision,
                "overall_recall": cat_overall_recall,
                "overall_f1": cat_overall_f1,
                "total_true_positives": cat_total_tp,
                "total_false_positives": cat_total_fp,
                "total_false_negatives": cat_total_fn
            },
            "deficiency_level": {
                "average_precision": def_total_precision,
                "average_recall": def_total_recall,
                "average_f1": def_total_f1,
                "overall_precision": def_overall_precision,
                "overall_recall": def_overall_recall,
                "overall_f1": def_overall_f1,
                "total_true_positives": def_total_tp,
                "total_false_positives": def_total_fp,
                "total_false_negatives": def_total_fn
            }
        },
        "per_class_metrics": per_class_results,
        "detailed_results": all_results
    }

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(all_results)}")
    
    print(f"\n{'='*30} CATEGORY-LEVEL METRICS {'='*30}")
    print(f"Average Precision: {cat_total_precision:.4f}")
    print(f"Average Recall: {cat_total_recall:.4f}")
    print(f"Average F1: {cat_total_f1:.4f}")
    print(f"Overall Precision: {cat_overall_precision:.4f}")
    print(f"Overall Recall: {cat_overall_recall:.4f}")
    print(f"Overall F1: {cat_overall_f1:.4f}")
    
    print(f"\nPer-Category Performance:")
    for cat_name, metrics in per_class_results["category_metrics"].items():
        if metrics["support"] > 0:
            print(f"  {cat_name:30s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}, Support: {metrics['support']}")
    
    print(f"\n{'='*30} DEFICIENCY-LEVEL METRICS {'='*30}")
    print(f"Average Precision: {def_total_precision:.4f}")
    print(f"Average Recall: {def_total_recall:.4f}")
    print(f"Average F1: {def_total_f1:.4f}")
    print(f"Overall Precision: {def_overall_precision:.4f}")
    print(f"Overall Recall: {def_overall_recall:.4f}")
    print(f"Overall F1: {def_overall_f1:.4f}")
    
    print(f"\nPer-Deficiency Performance (showing deficiencies with support > 0):")
    for def_name, metrics in per_class_results["deficiency_metrics"].items():
        if metrics["support"] > 0:
            print(f"  {def_name:50s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}, Support: {metrics['support']}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()