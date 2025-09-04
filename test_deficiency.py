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

class CategoryBooleans(BaseModel):
    composition_layout: bool
    typography: bool
    color: bool
    imagery_visualizations: bool


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


def load_classify_categories_prompt() -> str:
    """Load the classify categories prompt from file."""
    prompt_file = "src/open-r1-multimodal/prompts/classify_categories_prompt.txt"
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load prompt from {prompt_file}: {e}")
    return ""


def extract_answer_content(text: str) -> str:
    """Extract content from <answer></answer> tags"""
    import re
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class CategoryBooleanResponse(BaseModel):
    composition_layout: bool
    typography: bool
    color: bool
    imagery_visualizations: bool


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


def classify_categories(model_output_text: str) -> List[str]:
    """
    Uses OpenAI API to classify whether each main category has deficiencies (booleans).
    Returns a list of category names predicted as True.
    """
    input_text = extract_answer_content(model_output_text)
    main_categories = list(DEFICIENCY_CATEGORIES.keys())

    # Load prompt from file
    prompt_template = load_classify_categories_prompt()
    prompt = prompt_template.format(
        categories=json.dumps(main_categories, indent=2),
        input_text=input_text
    )

    try:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL")
        )

        model_type = os.getenv("MODEL_TYPE", "gpt-4o-2024-08-06")

        completion = client.chat.completions.create(
            model=model_type,
            messages=[
                {"role": "system", "content": "You are an expert in slide design analysis. Your output must be a valid JSON object with booleans per category."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        response_text = completion.choices[0].message.content
        if not response_text:
            return []

        parsed = CategoryBooleanResponse.model_validate_json(response_text)
        bools = {
            "Composition & Layout": getattr(parsed, "composition_layout", False),
            "Typography": getattr(parsed, "typography", False),
            "Color": getattr(parsed, "color", False),
            "Imagery & Visualizations": getattr(parsed, "imagery_visualizations", False),
        }
        return [name for name, flag in bools.items() if flag]

    except Exception as e:
        print(f"Error calling OpenAI API or parsing response: {e}")
        return []


class DeficiencyTester:
    def _load_prompt_from_file(self, prompt_file: str, default_prompt: str = "") -> str:
        """Load prompt text from file, with fallback to default if file doesn't exist."""
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not load prompt from {prompt_file}: {e}")
        return default_prompt


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

        # Load prompts from files
        self.system_prompt = self._load_prompt_from_file("src/open-r1-multimodal/prompts/system_prompt.txt")
        self.deficiency_prompt = self._load_prompt_from_file("src/open-r1-multimodal/prompts/deficiency_prompt.txt")

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
            
            # Predict major categories directly via boolean classification
            predicted_categories = set(classify_categories(generated_text))

            # Get ground truth specific deficiencies and map to categories
            ground_truth_deficiencies = [d["deficiency"] for d in item["deficiencies"]]
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

            result = {
                "slide_id": item["slide_id"],
                "image": item["image"],
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
            }

            print(f"Processed slide {item['slide_id']} - Cat F1: {cat_f1:.3f}")
            return result

        except Exception as e:
            print(f"Error processing slide {item['slide_id']}: {e}")
            return None


def calculate_per_class_metrics(results: List[Dict]) -> Dict:
    """Calculate per-category metrics only."""
    # Initialize counters for each category
    category_stats = {cat: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cat in DEFICIENCY_CATEGORIES.keys()}
    
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
    
    return {
        "category_metrics": category_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Test deficiency detection with API")
    parser.add_argument("--test_data", type=str, default="slideaudit_test.json",
                        help="Path to test data JSON file")
    parser.add_argument("--output_file", type=str, default="deficiency_result/eval_deficiency_f1_0.6_ep_1_new.json",
                        help="Output file for results")
    parser.add_argument("--num_workers", type=int, default=30,
                        help="Number of concurrent API workers")
    parser.add_argument("--report", action="store_true",
                        help="If set, skip API calls and report from --output_file JSON")

    args = parser.parse_args()

    # If only reporting is requested, skip API calls and read existing results
    if args.report:
        results_path = args.output_file
        if not results_path or not os.path.exists(results_path):
            raise FileNotFoundError("In --report mode, provide an existing --output_file JSON to read")

        with open(results_path, 'r') as f:
            prev_results = json.load(f)

        all_results = prev_results.get("detailed_results", [])

        # Calculate per-class metrics and distribution
        if all_results:
            per_class_results = calculate_per_class_metrics(all_results)
            category_distribution = {}
            total_samples = len(all_results)
            for cat_name in DEFICIENCY_CATEGORIES.keys():
                count = sum(1 for r in all_results if cat_name in r.get("predicted_categories", []))
                category_distribution[cat_name] = {
                    "count": count,
                    "rate": (count / total_samples) if total_samples > 0 else 0.0
                }

            print(f"\n{'='*60}")
            print(f"REPORT SUMMARY (from {results_path})")
            print(f"{'='*60}")
            print(f"Total samples (in results): {len(all_results)}")

            print(f"\n{'='*30} CATEGORY-LEVEL METRICS {'='*30}")
            for cat_name, metrics in per_class_results["category_metrics"].items():
                if metrics["support"] > 0:
                    print(f"  {cat_name:30s} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}, Support: {metrics['support']}")

            print(f"\n{'='*30} CATEGORY DISTRIBUTION {'='*30}")
            for cat_name, stats in category_distribution.items():
                print(f"  {cat_name:30s} - count: {stats['count']}, rate: {stats['rate']:.3f}")

        else:
            print("No detailed_results found in the results file.")
        return

    # Load test data (normal run)
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

        # Calculate per-class metrics
        per_class_results = calculate_per_class_metrics(all_results)
        
        # Optional: Category prediction distribution across all samples
        category_distribution = {}
        total_samples = len(all_results)
        for cat_name in DEFICIENCY_CATEGORIES.keys():
            count = sum(1 for r in all_results if cat_name in r["predicted_categories"])
            category_distribution[cat_name] = {
                "count": count,
                "rate": (count / total_samples) if total_samples > 0 else 0.0
            }
        
    else:
        cat_total_precision = cat_total_recall = cat_total_f1 = 0
        cat_overall_precision = cat_overall_recall = cat_overall_f1 = 0
        cat_total_tp = cat_total_fp = cat_total_fn = 0
        
        per_class_results = {"category_metrics": {}}
        category_distribution = {}

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
        },
        "per_class_metrics": per_class_results,
        "category_distribution": category_distribution if args.report else {},
        "detailed_results": all_results
    }

    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
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
    
    # Optional distribution report
    if args.report and category_distribution:
        print(f"\n{'='*30} CATEGORY DISTRIBUTION {'='*30}")
        for cat_name, stats in category_distribution.items():
            print(f"  {cat_name:30s} - count: {stats['count']}, rate: {stats['rate']:.3f}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()