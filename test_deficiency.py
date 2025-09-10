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
        "Content Alignment Issues",
        "Content Overflow/Cut-off",
        "Unbalanced Space Distribution"
    ],
    "Typography": [
        "Illegible Typeface Selection or Usage",
        "Improper Font Sizing",
        "Excessive Text Volume",
        "Improper Line/Character Spacing"
    ],
    "Imagery & Visualizations": [
        "Irrelevant Visual Content",
        "Improper Image Sizing",
        "Inconsistent Visual Style Usage",
        "Inappropriate or Mismatched Color Combinations"
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
    prompt_file = "train/prompts/classify_categories_prompt.txt"
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
            "Imagery & Visualizations": getattr(parsed, "imagery_visualizations", False),
        }
        return [name for name, flag in bools.items() if flag]

    except Exception as e:
        print(f"Error calling OpenAI API or parsing response: {e}")
        return []


class MultiTaskTester:
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
        self.system_prompt = self._load_prompt_from_file("train/prompts/system_prompt.txt")
        self.deficiency_prompt = self._load_prompt_from_file("train/prompts/deficiency_prompt.txt")
        self.score_prompt = self._load_prompt_from_file("train/prompts/score_question_prompt.txt")

    def process_deficiency(self, item: Dict, max_retries: int = 3) -> Dict:
        """Process a single deficiency test sample: call API, classify deficiencies, and calculate metrics."""
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
                    print(f"API call failed for slide {item.get('slide_id', 'unknown')} (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"Failed after {max_retries} attempts for slide {item.get('slide_id', 'unknown')}")
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
                "task": "deficiency",
                "slide_id": item.get("slide_id", "unknown"),
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

            print(f"Processed deficiency slide {item.get('slide_id', 'unknown')} - Cat F1: {cat_f1:.3f}")
            return result

        except Exception as e:
            print(f"Error processing deficiency slide {item.get('slide_id', 'unknown')}: {e}")
            return None

    def process_score(self, item: Dict, max_retries: int = 3) -> Dict:
        """Process a single score test sample: call API, extract score, and calculate metrics."""
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
                        {"type": "text", "text": self.score_prompt},
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
                    print(f"API call failed for slide {item.get('slide_id', 'unknown')} (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"Failed after {max_retries} attempts for slide {item.get('slide_id', 'unknown')}")
                        return None
            
            # Extract score from <answer> tags
            answer_content = extract_answer_content(generated_text)
            import re
            score_match = re.search(r'(\d+\.?\d*)', answer_content)
            predicted_score = float(score_match.group(1)) if score_match else None

            # Get ground truth score (overall if dict, otherwise direct value)
            gt_score = item.get("score", None) or item.get("gt_score_norm", None)
            if isinstance(gt_score, dict):
                gt_score = gt_score.get("overall")

            # Calculate metrics
            if predicted_score is not None and gt_score is not None:
                mae = abs(predicted_score - gt_score)
                mse = (predicted_score - gt_score) ** 2
                # Threshold-based accuracy (within 0.35 as in training)
                threshold = 0.35
                within_threshold = 1.0 if mae < threshold else 0.0
            else:
                mae = mse = within_threshold = None

            result = {
                "task": "score",
                "slide_id": item.get("slide_id", "unknown"),
                "image": item["image"],
                "ground_truth_score": gt_score,
                "predicted_score": predicted_score,
                "model_output": generated_text,
                "metrics": {
                    "mae": mae,
                    "mse": mse,
                    "within_threshold": within_threshold,
                    "threshold": 0.35
                },
            }

            print(f"Processed score slide {item.get('slide_id', 'unknown')} - Pred: {predicted_score}, GT: {gt_score}, MAE: {mae:.3f}")
            return result

        except Exception as e:
            print(f"Error processing score slide {item.get('slide_id', 'unknown')}: {e}")
            return None

    def process_single(self, item: Dict, task_type: str = "deficiency", max_retries: int = 3) -> Dict:
        """Process a single test sample based on task type."""
        if task_type == "deficiency":
            return self.process_deficiency(item, max_retries)
        elif task_type == "score":
            return self.process_score(item, max_retries)
        else:
            raise ValueError(f"Unknown task type: {task_type}")


def calculate_per_class_metrics(results: List[Dict]) -> Dict:
    """Calculate per-category metrics for deficiency tasks only."""
    # Filter deficiency results
    deficiency_results = [r for r in results if r.get("task") == "deficiency"]
    
    # Initialize counters for each category
    category_stats = {cat: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for cat in DEFICIENCY_CATEGORIES.keys()}
    
    for result in deficiency_results:
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


def calculate_score_metrics(results: List[Dict]) -> Dict:
    """Calculate score prediction metrics."""
    # Filter score results
    score_results = [r for r in results if r.get("task") == "score"]
    
    if not score_results:
        return {"score_metrics": {}}
    
    # Calculate overall metrics
    valid_scores = [(r["predicted_score"], r["ground_truth_score"]) for r in score_results 
                   if r["predicted_score"] is not None and r["ground_truth_score"] is not None]
    
    if not valid_scores:
        return {"score_metrics": {"error": "No valid score predictions found"}}
    
    predicted_scores, gt_scores = zip(*valid_scores)
    
    # Calculate metrics
    mae = sum(abs(p - gt) for p, gt in valid_scores) / len(valid_scores)
    mse = sum((p - gt) ** 2 for p, gt in valid_scores) / len(valid_scores)
    rmse = mse ** 0.5
    
    # Threshold-based accuracy
    threshold = 0.35
    within_threshold = sum(1 for p, gt in valid_scores if abs(p - gt) < threshold) / len(valid_scores)
    
    return {
        "score_metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "within_threshold": within_threshold,
            "threshold": threshold,
            "total_samples": len(valid_scores),
            "valid_predictions": len(valid_scores),
            "failed_predictions": len(score_results) - len(valid_scores)
        }
    }


def print_results(results: Dict, task_type: str):
    """Print results summary for a specific task."""
    all_results = results.get("detailed_results", [])
    overall_metrics = results.get("overall_metrics", {})
    per_class_metrics = results.get("per_class_metrics", {})
    category_distribution = results.get("category_distribution", {})
    
    print(f"Total samples processed: {len(all_results)}")
    
    if task_type == "deficiency" and "deficiency" in overall_metrics:
        metrics = overall_metrics["deficiency"]["category_level"]
        print(f"\n{'='*30} CATEGORY-LEVEL METRICS {'='*30}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"Average Recall: {metrics['average_recall']:.4f}")
        print(f"Average F1: {metrics['average_f1']:.4f}")
        print(f"Overall Precision: {metrics['overall_precision']:.4f}")
        print(f"Overall Recall: {metrics['overall_recall']:.4f}")
        print(f"Overall F1: {metrics['overall_f1']:.4f}")
        
        print(f"\nPer-Category Performance:")
        for cat_name, cat_metrics in per_class_metrics.get("category_metrics", {}).items():
            if cat_metrics["support"] > 0:
                print(f"  {cat_name:30s} - P: {cat_metrics['precision']:.3f}, R: {cat_metrics['recall']:.3f}, F1: {cat_metrics['f1']:.3f}, Support: {cat_metrics['support']}")
        
        if category_distribution:
            print(f"\n{'='*30} CATEGORY DISTRIBUTION {'='*30}")
            for cat_name, stats in category_distribution.items():
                print(f"  {cat_name:30s} - count: {stats['count']}, rate: {stats['rate']:.3f}")
                
    elif task_type == "score" and "score" in overall_metrics:
        metrics = overall_metrics["score"]
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
        else:
            print(f"\n{'='*30} SCORE PREDICTION METRICS {'='*30}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  Within Threshold ({metrics['threshold']}): {metrics['within_threshold']:.4f}")
            print(f"  Valid Predictions: {metrics['valid_predictions']}")
            print(f"  Failed Predictions: {metrics['failed_predictions']}")


def process_task(tester: MultiTaskTester, test_data_file: str, task_type: str, num_workers: int) -> List[Dict]:
    """Process a single task with given test data."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {task_type.upper()} TASK")
    print(f"{'='*60}")
    
    # Load test data
    with open(test_data_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} {task_type} test samples")
    print(f"Using {num_workers} concurrent workers")
    
    all_results = []
    
    if num_workers == 1:
        # Single thread mode
        for item in tqdm(test_data, desc=f"Processing {task_type} samples"):
            result = tester.process_single(item, task_type=task_type)
            if result:
                all_results.append(result)
    else:
        # Multi-threaded mode for concurrent API calls
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(tester.process_single, item, task_type): item for item in test_data}
            
            # Collect results with progress bar
            with tqdm(total=len(test_data), desc=f"Processing {task_type} samples") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        all_results.append(result)
                    pbar.update(1)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Test multi-task model with API")
    parser.add_argument("--deficiency_test_data", type=str, default="dataset/deficiency_test.json",
                        help="Path to deficiency test data JSON file")
    parser.add_argument("--score_test_data", type=str, default="dataset/score_test.json",
                        help="Path to score test data JSON file")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Output directory for results")
    parser.add_argument("--num_workers", type=int, default=30,
                        help="Number of concurrent API workers")
    parser.add_argument("--task", type=str, choices=["deficiency", "score", "both"], default="both",
                        help="Task type to test: deficiency, score, or both")
    parser.add_argument("--report", action="store_true",
                        help="If set, skip API calls and report from existing results")

    args = parser.parse_args()

    # If only reporting is requested, skip API calls and read existing results
    if args.report:
        # Extract model name from TEST_MODEL path for filename
        test_model_path = os.getenv("TEST_MODEL", "unknown_model")
        model_name = os.path.basename(test_model_path) if test_model_path != "unknown_model" else "unknown_model"
        
        if args.task == "both":
            # Report both tasks
            deficiency_file = os.path.join(args.output_dir, f"deficiency_{model_name}.json")
            score_file = os.path.join(args.output_dir, f"score_{model_name}.json")
            
            if os.path.exists(deficiency_file):
                print(f"\n{'='*60}")
                print(f"DEFICIENCY REPORT SUMMARY")
                print(f"{'='*60}")
                with open(deficiency_file, 'r') as f:
                    deficiency_results = json.load(f)
                print_results(deficiency_results, "deficiency")
            
            if os.path.exists(score_file):
                print(f"\n{'='*60}")
                print(f"SCORE REPORT SUMMARY")
                print(f"{'='*60}")
                with open(score_file, 'r') as f:
                    score_results = json.load(f)
                print_results(score_results, "score")
        else:
            # Report single task
            output_file = os.path.join(args.output_dir, f"{args.task}_{model_name}.json")
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Results file not found: {output_file}")
            
            with open(output_file, 'r') as f:
                results = json.load(f)
            print_results(results, args.task)
        return

    # Initialize tester
    print(f"Test Model: {os.getenv('TEST_MODEL')}")
    print(f"API Base URL: {os.getenv('TEST_BASE_URL')}")
    tester = MultiTaskTester()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which tasks to run
    tasks_to_run = []
    if args.task == "both":
        tasks_to_run = ["deficiency", "score"]
    else:
        tasks_to_run = [args.task]
    
    # Process each task
    for task_type in tasks_to_run:
        if task_type == "deficiency":
            test_data_file = args.deficiency_test_data
        elif task_type == "score":
            test_data_file = args.score_test_data
        else:
            continue
            
        # Check if test data file exists
        if not os.path.exists(test_data_file):
            print(f"Warning: Test data file not found: {test_data_file}")
            continue
            
        # Process the task
        all_results = process_task(tester, test_data_file, task_type, args.num_workers)
        
        # Calculate metrics based on task type
        if all_results:
            if task_type == "deficiency":
                # Deficiency-specific metrics
                deficiency_results = [r for r in all_results if r.get("task") == "deficiency"]
                if deficiency_results:
                    cat_total_precision = sum(r["category_metrics"]["precision"] for r in deficiency_results) / len(deficiency_results)
                    cat_total_recall = sum(r["category_metrics"]["recall"] for r in deficiency_results) / len(deficiency_results)
                    cat_total_f1 = sum(r["category_metrics"]["f1"] for r in deficiency_results) / len(deficiency_results)

                    cat_total_tp = sum(r["category_metrics"]["true_positives"] for r in deficiency_results)
                    cat_total_fp = sum(r["category_metrics"]["false_positives"] for r in deficiency_results)
                    cat_total_fn = sum(r["category_metrics"]["false_negatives"] for r in deficiency_results)

                    cat_overall_precision = cat_total_tp / (cat_total_tp + cat_total_fp) if (cat_total_tp + cat_total_fp) > 0 else 0
                    cat_overall_recall = cat_total_tp / (cat_total_tp + cat_total_fn) if (cat_total_tp + cat_total_fn) > 0 else 0
                    cat_overall_f1 = 2 * cat_overall_precision * cat_overall_recall / (cat_overall_precision + cat_overall_recall) if (cat_overall_precision + cat_overall_recall) > 0 else 0
                else:
                    cat_total_precision = cat_total_recall = cat_total_f1 = 0
                    cat_overall_precision = cat_overall_recall = cat_overall_f1 = 0
                    cat_total_tp = cat_total_fp = cat_total_fn = 0

                per_class_results = calculate_per_class_metrics(all_results)
                
                # Category prediction distribution
                category_distribution = {}
                total_samples = len(deficiency_results)
                for cat_name in DEFICIENCY_CATEGORIES.keys():
                    count = sum(1 for r in deficiency_results if cat_name in r.get("predicted_categories", []))
                    category_distribution[cat_name] = {
                        "count": count,
                        "rate": (count / total_samples) if total_samples > 0 else 0.0
                    }
                
                overall_metrics = {
                    "deficiency": {
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
                    }
                }
                
            elif task_type == "score":
                # Score-specific metrics
                score_results = calculate_score_metrics(all_results)
                overall_metrics = {
                    "score": score_results["score_metrics"]
                }
                per_class_results = {}
                category_distribution = {}
            
        else:
            overall_metrics = {}
            per_class_results = {}
            category_distribution = {}

        # Prepare final results
        final_results = {
            "task": task_type,
            "test_model": os.getenv("TEST_MODEL"),
            "test_api_base_url": os.getenv("TEST_BASE_URL"),
            "test_data": test_data_file,
            "total_samples": len(all_results),
            "overall_metrics": overall_metrics,
            "per_class_metrics": per_class_results,
            "category_distribution": category_distribution,
            "detailed_results": all_results
        }

        # Extract model name from TEST_MODEL path for filename
        test_model_path = os.getenv("TEST_MODEL", "unknown_model")
        model_name = os.path.basename(test_model_path) if test_model_path != "unknown_model" else "unknown_model"
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{task_type}_{model_name}.json")
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # Print summary
        print_results(final_results, task_type)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"ALL TASKS COMPLETED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()