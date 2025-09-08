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
import time

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

def load_prompt_from_file(prompt_file: str, default_prompt: str = "") -> str:
    """Load prompt text from file, with fallback to default if file doesn't exist."""
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load prompt from {prompt_file}: {e}")
    return default_prompt

def extract_answer_content(text: str) -> str:
    """Extract content from <answer></answer> tags"""
    import re
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def image_to_base64_uri(image_path: str, max_width: int = 720) -> str:
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

def normalize_prediction(prediction: str) -> str:
    """Normalize prediction to one of: Slide A, Slide B, or Similar"""
    prediction = prediction.strip().lower()
    
    # Check for various ways to express the same choice
    if any(phrase in prediction for phrase in ['slide a', 'image a', 'a is', 'a better', 'a superior']):
        return "Slide A"
    elif any(phrase in prediction for phrase in ['slide b', 'image b', 'b is', 'b better', 'b superior']):
        return "Slide B"
    elif any(phrase in prediction for phrase in ['similar', 'comparable', 'equal', 'same', 'tie']):
        return "Similar"
    else:
        # Default fallback - try to extract from the text
        if 'a' in prediction and 'b' not in prediction:
            return "Slide A"
        elif 'b' in prediction and 'a' not in prediction:
            return "Slide B"
        else:
            return "Similar"

class CompareTester:
    def __init__(self, image_root: str = "dataset/compare"):
        # Load API configuration
        self.test_model = os.getenv("TEST_MODEL")
        self.test_api_key = os.getenv("TEST_API_KEY")
        self.test_base_url = os.getenv("TEST_BASE_URL")
        self.image_root = image_root
        
        if not all([self.test_model, self.test_api_key, self.test_base_url]):
            raise ValueError("TEST_MODEL, TEST_API_KEY, and TEST_BASE_URL must be set in .env file")
        
        # Initialize OpenAI client for test model
        self.client = openai.OpenAI(
            api_key=self.test_api_key,
            base_url=self.test_base_url
        )

        # Load prompts from files
        self.system_prompt = load_prompt_from_file("src/open-r1-multimodal/prompts/system_prompt.txt")
        self.compare_prompt = load_prompt_from_file("src/open-r1-multimodal/prompts/compare_question_prompt.txt")

    def process_single(self, item: Dict, max_retries: int = 3) -> Dict:
        """Process a single test sample: call API and calculate metrics."""
        try:
            # Convert images to base64 with 720 pixel width constraint
            ref_image_path = os.path.join(self.image_root, item["ref_image"])
            image_a_path = os.path.join(self.image_root, item["ImageA"])
            image_b_path = os.path.join(self.image_root, item["ImageB"])
            
            ref_image_uri = image_to_base64_uri(ref_image_path, max_width=720)
            image_a_uri = image_to_base64_uri(image_a_path, max_width=720)
            image_b_uri = image_to_base64_uri(image_b_path, max_width=720)
            
            if not all([ref_image_uri, image_a_uri, image_b_uri]):
                return None

            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.compare_prompt},
                        {"type": "image_url", "image_url": {"url": ref_image_uri}},
                        {"type": "image_url", "image_url": {"url": image_a_uri}},
                        {"type": "image_url", "image_url": {"url": image_b_uri}}
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
                    print(f"API call failed for sample (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        print(f"Failed after {max_retries} attempts")
                        return None
            
            # Extract and normalize prediction
            raw_prediction = extract_answer_content(generated_text)
            predicted_result = normalize_prediction(raw_prediction)
            
            # Get ground truth
            ground_truth = item["result"]
            
            # Calculate accuracy
            is_correct = predicted_result == ground_truth

            result = {
                "ref_image": item["ref_image"],
                "ImageA": item["ImageA"],
                "ImageB": item["ImageB"],
                "ground_truth": ground_truth,
                "predicted_result": predicted_result,
                "raw_prediction": raw_prediction,
                "model_output": generated_text,
                "is_correct": is_correct
            }

            print(f"Processed sample - GT: {ground_truth}, Pred: {predicted_result}, Correct: {is_correct}")
            return result

        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate overall metrics for comparison task."""
    if not results:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "correct_predictions": 0,
            "incorrect_predictions": 0
        }
    
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])
    incorrect_predictions = total_samples - correct_predictions
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Calculate per-class metrics
    class_stats = {"Slide A": {"correct": 0, "total": 0}, 
                   "Slide B": {"correct": 0, "total": 0}, 
                   "Similar": {"correct": 0, "total": 0}}
    
    for result in results:
        gt = result["ground_truth"]
        is_correct = result["is_correct"]
        
        if gt in class_stats:
            class_stats[gt]["total"] += 1
            if is_correct:
                class_stats[gt]["correct"] += 1
    
    per_class_metrics = {}
    for class_name, stats in class_stats.items():
        if stats["total"] > 0:
            per_class_metrics[class_name] = {
                "accuracy": stats["correct"] / stats["total"],
                "support": stats["total"],
                "correct": stats["correct"]
            }
        else:
            per_class_metrics[class_name] = {
                "accuracy": 0.0,
                "support": 0,
                "correct": 0
            }
    
    return {
        "total_samples": total_samples,
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "per_class_metrics": per_class_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Test comparison task with API")
    parser.add_argument("--test_data", type=str, default="dataset/compare/test_comparison.json",
                        help="Path to test data JSON file")
    parser.add_argument("--output_file", type=str, default="compare_result/eval_compare_ep3_dp1500.json",
                        help="Output file for results")
    parser.add_argument("--num_workers", type=int, default=50,
                        help="Number of concurrent API workers")
    parser.add_argument("--image_root", type=str, default="dataset/compare",
                        help="Root directory for images")
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

        # Calculate metrics
        if all_results:
            metrics = calculate_metrics(all_results)

            print(f"\n{'='*60}")
            print(f"REPORT SUMMARY (from {results_path})")
            print(f"{'='*60}")
            print(f"Total samples: {metrics['total_samples']}")
            print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
            print(f"Correct predictions: {metrics['correct_predictions']}")
            print(f"Incorrect predictions: {metrics['incorrect_predictions']}")

            print(f"\n{'='*30} PER-CLASS METRICS {'='*30}")
            for class_name, class_metrics in metrics["per_class_metrics"].items():
                if class_metrics["support"] > 0:
                    print(f"  {class_name:10s} - Accuracy: {class_metrics['accuracy']:.3f}, Support: {class_metrics['support']}")

        else:
            print("No detailed_results found in the results file.")
        return

    # Load test data (normal run)
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test samples")
    print(f"Using {args.num_workers} concurrent workers")
    print(f"Image root: {args.image_root}")
    print(f"Test Model: {os.getenv('TEST_MODEL')}")
    print(f"API Base URL: {os.getenv('TEST_BASE_URL')}")

    # Initialize tester
    tester = CompareTester(image_root=args.image_root)
    
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

    # Calculate metrics
    metrics = calculate_metrics(all_results)

    # Prepare final results
    final_results = {
        "test_model": os.getenv("TEST_MODEL"),
        "test_api_base_url": os.getenv("TEST_BASE_URL"),
        "test_data": args.test_data,
        "image_root": args.image_root,
        "metrics": metrics,
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
    print(f"Total samples processed: {metrics['total_samples']}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Incorrect predictions: {metrics['incorrect_predictions']}")
    
    print(f"\nPer-Class Performance:")
    for class_name, class_metrics in metrics["per_class_metrics"].items():
        if class_metrics["support"] > 0:
            print(f"  {class_name:10s} - Accuracy: {class_metrics['accuracy']:.3f}, Support: {class_metrics['support']}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
