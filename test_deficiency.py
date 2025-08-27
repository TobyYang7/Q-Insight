import json
import os
import base64
import io
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count, set_start_method
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info
import openai
from pydantic import BaseModel

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
    all_deficiencies = list(DEFICIENCY_TO_CATEGORY_MAP.keys())

    prompt = f"""Analyze the input text which describes slide design problems. From the predefined categories, identify all deficiencies mentioned in the text.

    Predefined deficiency categories:
    {json.dumps(all_deficiencies, indent=2)}

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
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device

        set_seed(42)

        self.system_prompt = (
            "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
            "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think><answer> answer here </answer>"
        )

        self.deficiency_prompt = (
            "Please provide a professional design critique of the accompanying slide. "
            "If there are no deficiencies, you should say 'No deficiencies'."
            "Otherwise, your analysis should identify any design deficiencies, explain the reasoning behind your critique, "
            "and offer specific, actionable suggestions for improvement. "
        )

        # Initialize model and processor
        print(f"Loading model on device: {self.device}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        # Generation config
        self.gen_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            max_new_tokens=1024,
        )

    def process_single_image(self, image_path: str) -> str:
        """Process a single image and return model output"""
        base64_image_uri = image_to_base64_uri(image_path)
        if not base64_image_uri:
            return ""

        message = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.deficiency_prompt},
                    {"type": "image", "image": base64_image_uri}
                ]
            }
        ]

        text = [self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info([message])
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            generation_config=self.gen_config,
            use_cache=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

    def process_single_sample(self, item: Dict) -> Dict:
        """Process a single test sample and calculate metrics based on major categories."""
        try:
            # Run GPU inference to get model output
            generated_text = self.process_single_image(item["image"])
            # Classify specific deficiencies from the generated text
            predicted_deficiencies = classify_deficiencies(generated_text)

            # Get ground truth specific deficiencies
            ground_truth_deficiencies = [d["deficiency"] for d in item["deficiencies"]]

            # --- MODIFICATION START: Convert specific deficiencies to major categories ---
            predicted_categories = {
                DEFICIENCY_TO_CATEGORY_MAP[d] for d in predicted_deficiencies if d in DEFICIENCY_TO_CATEGORY_MAP
            }
            ground_truth_categories = {
                DEFICIENCY_TO_CATEGORY_MAP[d] for d in ground_truth_deficiencies if d in DEFICIENCY_TO_CATEGORY_MAP
            }
            # --- MODIFICATION END ---

            # --- MODIFICATION: Calculate accuracy metrics based on CATEGORIES ---
            true_positives = len(predicted_categories.intersection(ground_truth_categories))
            false_positives = len(predicted_categories - ground_truth_categories)
            false_negatives = len(ground_truth_categories - predicted_categories)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            result = {
                "slide_id": item["slide_id"],
                "image": item["image"],
                "ground_truth_deficiencies": ground_truth_deficiencies,
                "predicted_deficiencies": predicted_deficiencies,
                "ground_truth_categories": sorted(list(ground_truth_categories)),  # Added for clarity
                "predicted_categories": sorted(list(predicted_categories)),     # Added for clarity
                "model_output": generated_text,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }

            print(f"Processed slide {item['slide_id']} on {self.device}")
            return result

        except Exception as e:
            print(f"Error processing slide {item['slide_id']}: {e}")
            return None


def worker_process(args):
    """Worker function for multiprocessing - load model once per GPU, process assigned samples"""
    samples_for_gpu, model_path, gpu_id = args
    device = f"cuda:{gpu_id}"

    # Load model once per GPU
    print(f"Loading model on {device}")
    tester = DeficiencyTester(model_path, device)

    # Process all samples assigned to this GPU
    results = []
    for sample in samples_for_gpu:
        result = tester.process_single_sample(sample)
        if result:
            results.append(result)

    return results


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description="Test deficiency detection with multi-GPU")
    parser.add_argument("--model_path", type=str, default="model/VLAA-Thinker-Qwen2.5VL-7B",
                        help="Path to the model")
    parser.add_argument("--test_data", type=str, default="dataset/slideaudit_test.json",
                        help="Path to test data JSON file")
    parser.add_argument("--output_file", type=str, default="deficiency_test_results_VLAA-Thinker-Qwen2.5VL-7B.json",
                        help="Output file for results")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: number of available GPUs)")

    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Determine number of workers
    num_gpus = torch.cuda.device_count()
    if args.num_workers is None:
        args.num_workers = num_gpus
    else:
        args.num_workers = min(args.num_workers, num_gpus)

    if args.num_workers == 0:
        print("Error: No available GPUs found. This script requires at least one GPU.")
        return

    print(f"Loaded {len(test_data)} test samples")
    print(f"Available GPUs: {num_gpus}")
    print(f"Using {args.num_workers} worker processes")

    all_results = []

    if args.num_workers == 1:
        # Single process mode
        device = "cuda:0"
        tester = DeficiencyTester(args.model_path, device)
        for item in tqdm(test_data, desc="Processing samples"):
            result = tester.process_single_sample(item)
            if result:
                all_results.append(result)
    else:
        # Multi-process mode - divide samples among GPUs
        samples_per_gpu = len(test_data) // args.num_workers
        remainder = len(test_data) % args.num_workers

        worker_args = []
        start_idx = 0

        for gpu_id in range(args.num_workers):
            num_samples = samples_per_gpu + (1 if gpu_id < remainder else 0)
            end_idx = start_idx + num_samples

            samples_for_gpu = test_data[start_idx:end_idx]
            worker_args.append((samples_for_gpu, args.model_path, gpu_id))

            print(f"GPU {gpu_id} will process {len(samples_for_gpu)} samples")
            start_idx = end_idx

        with Pool(processes=args.num_workers) as pool:
            gpu_results = pool.map(worker_process, worker_args)

            # Flatten results from all GPUs
            all_results = []
            for gpu_result in gpu_results:
                all_results.extend(gpu_result)

    # Calculate overall metrics
    total_precision = sum(r["precision"] for r in all_results) / len(all_results) if all_results else 0
    total_recall = sum(r["recall"] for r in all_results) / len(all_results) if all_results else 0
    total_f1 = sum(r["f1"] for r in all_results) / len(all_results) if all_results else 0

    total_tp = sum(r["true_positives"] for r in all_results)
    total_fp = sum(r["false_positives"] for r in all_results)
    total_fn = sum(r["false_negatives"] for r in all_results)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    # Prepare final results
    final_results = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "total_samples": len(all_results),
        "overall_metrics": {
            "average_precision": total_precision,
            "average_recall": total_recall,
            "average_f1": total_f1,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn
        },
        "detailed_results": all_results
    }

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Results Summary (Calculated by Category) ===")
    print(f"Total samples processed: {len(all_results)}")
    print(f"Average Precision: {total_precision:.4f}")
    print(f"Average Recall: {total_recall:.4f}")
    print(f"Average F1: {total_f1:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
