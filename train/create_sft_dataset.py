#!/usr/bin/env python3
"""
Script to create SFT dataset in the specified format from the training data.
This script reads the same data sources as train_sft.py and converts them to the required format.
"""

import os
import json
import yaml
import math
import random
import argparse
from typing import Dict, List, Any, Optional
from PIL import Image


def load_prompt_from_file(prompt_file: str, default_prompt: str = "") -> str:
    """Load prompt from file or return default."""
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return default_prompt


# Load prompts (same as train_sft.py)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPTS_DIR, "system_prompt.txt")
DEFICIENCY_PROMPT_FILE = os.path.join(PROMPTS_DIR, "deficiency_prompt.txt")
SCORE_QUESTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "score_question_prompt.txt")
COMPARE_QUESTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "compare_question_prompt.txt")

SYSTEM_PROMPT = load_prompt_from_file(SYSTEM_PROMPT_FILE, "You are a helpful assistant specialized in slide design analysis.")
DEFICIENCY_PROMPT = load_prompt_from_file(DEFICIENCY_PROMPT_FILE, "Please identify any design deficiencies in this slide.")
SCORE_QUESTION_PROMPT = load_prompt_from_file(SCORE_QUESTION_PROMPT_FILE, "Please rate the quality of this slide on a scale of 1 to 5.")
COMPARE_QUESTION_PROMPT = load_prompt_from_file(COMPARE_QUESTION_PROMPT_FILE, "Compare Slide A and Slide B based on the reference slide. Which one is better?")


def load_samples_from_yaml(data_path: str, task_type: str = None):
    """Loads samples from a YAML config file that points to JSON data."""
    if not data_path or not os.path.exists(data_path):
        return []
    
    samples = []
    if not data_path.endswith(".yaml"):
        raise ValueError(f"Unsupported file type: {data_path}, must be a .yaml file.")
    
    with open(data_path, "r") as f:
        cfg = yaml.safe_load(f)
        
        # Check if this is a unified config format
        if task_type and task_type in cfg:
            # Load from unified config for specific task type
            task_config = cfg[task_type]
            if not isinstance(task_config, list):
                task_config = [task_config]
            
            for ds in task_config:
                samples.extend(_load_dataset_samples(ds))
        else:
            # Legacy format - load from "datasets" key
            for ds in cfg.get("datasets", []):
                samples.extend(_load_dataset_samples(ds))
    
    return samples


def _load_dataset_samples(ds):
    """Load samples from a single dataset configuration."""
    samples = []
    path = ds.get("json_path")
    strategy = ds.get("sampling_strategy", "all")
    image_root = ds.get("image_root")
    sample = ds.get("sample")  # New field for sample count
    random_sample = ds.get("random", False)  # New field for random sampling

    if path.endswith(".jsonl"):
        data_list = [json.loads(line) for line in open(path, "r")]
    elif path.endswith(".json"):
        data_list = json.load(open(path, "r"))
    else:
        raise ValueError(f"Unsupported file type: {path}")

    # Handle sampling strategy
    count = None
    if ":" in strategy:
        strat, num_s = strategy.split(":")
        if "%" in num_s:
            count = math.ceil(int(num_s.rstrip("%")) * len(data_list) / 100)
        else:
            count = int(num_s)
        strategy = strat
    elif sample is not None:
        # Use the new 'sample' field if available
        count = sample
        strategy = "random" if random_sample else "first"

    if strategy == "first" and count is not None:
        data_list = data_list[:count]
    elif strategy == "end" and count is not None:
        data_list = data_list[-count:]
    elif strategy == "random" and count is not None:
        random.shuffle(data_list)
        data_list = data_list[:count]

    for sample in data_list:
        sample['image_root'] = image_root

    samples.extend(data_list)
    return samples


def _format_deficiency_solution(deficiencies: List[Dict]) -> str:
    """Format deficiency analysis into the required response format."""
    if not deficiencies:
        return "<think>I have analyzed the slide and found no design deficiencies.</think><answer>No deficiencies found.</answer>"

    thought = f"I have analyzed the slide and found {len(deficiencies)} deficiency/deficiencies. I will list them now."
    answer_parts = [f"- **{d['category']}**: {d['deficiency']}" for d in deficiencies]
    answer = "\n".join(answer_parts)
    return f"<think>{thought}</think><answer>\n{answer}\n</answer>"


def _format_score_solution(score: Any) -> str:
    """Format score analysis into the required response format."""
    if isinstance(score, dict):
        score = score.get("overall", 0)
    
    try:
        score_val = float(score)
        thought = f"I have analyzed the slide and determined its quality score to be {score_val:.1f} out of 5."
        answer = f"{score_val:.1f}"
        return f"<think>{thought}</think><answer>{answer}</answer>"
    except (ValueError, TypeError):
        thought = "I was unable to determine a numerical score for the slide."
        answer = "Score not available."
        return f"<think>{thought}</think><answer>{answer}</answer>"


def _format_comparison_solution(result: str) -> str:
    """Format comparison analysis into the required response format."""
    thought = f"I have compared Slide A and Slide B against the reference. Based on my analysis, {result} is better."
    answer = result
    return f"<think>{thought}</think><answer>{answer}</answer>"


def convert_to_sft_format(samples: List[Dict], task_type: str) -> List[Dict]:
    """Convert samples to the required SFT format."""
    sft_samples = []
    
    for example in samples:
        image_root = example.get("image_root", "")
        
        if task_type == "comparison" or task_type == "compare":
            # Comparison task: 3 images (ref, A, B)
            ref_img_path = os.path.join(image_root, example["ref_image"])
            img_a_path = os.path.join(image_root, example["ImageA"])
            img_b_path = os.path.join(image_root, example["ImageB"])
            
            # Check if all images exist
            if not all(os.path.exists(p) for p in [ref_img_path, img_a_path, img_b_path]):
                print(f"Warning: Skipping comparison sample due to missing images: {example}")
                continue
            
            solution_text = _format_comparison_solution(example["result"])
            
            sft_sample = {
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image><image><image>{COMPARE_QUESTION_PROMPT}"
                    },
                    {
                        "role": "assistant", 
                        "content": solution_text
                    }
                ],
                "images": [ref_img_path, img_a_path, img_b_path]
            }
            
        else:  # Score or Deficiency
            img_key = example.get("image") or example.get("image_path")
            if not img_key:
                print(f"Warning: Skipping {task_type} sample due to missing image key: {example}")
                continue
            img_path = os.path.join(image_root, img_key)
            
            # Check if image exists
            if not os.path.exists(img_path):
                print(f"Warning: Skipping {task_type} sample due to missing image: {img_path}")
                continue
            
            if task_type == "score":
                prompt_text = SCORE_QUESTION_PROMPT
                solution = example.get("score") or example.get("gt_score_norm")
                solution_text = _format_score_solution(solution)
            else:  # deficiency
                prompt_text = DEFICIENCY_PROMPT
                solution_text = _format_deficiency_solution(example.get("deficiencies", []))
            
            sft_sample = {
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>{prompt_text}"
                    },
                    {
                        "role": "assistant",
                        "content": solution_text
                    }
                ],
                "images": [img_path]
            }
        
        sft_samples.append(sft_sample)
    
    return sft_samples


def main():
    parser = argparse.ArgumentParser(description="Create SFT dataset from training data")
    parser.add_argument("--dataset_config", type=str, required=True,
                       help="YAML file containing all dataset configurations")
    parser.add_argument("--output_file", type=str, default="sft_dataset.json",
                       help="Output file path for the SFT dataset")
    parser.add_argument("--task_types", nargs="+", 
                       choices=["score", "deficiency", "compare"],
                       default=["score", "deficiency", "compare"],
                       help="Task types to include in the dataset")
    
    args = parser.parse_args()
    
    print(f"Loading dataset from config: {args.dataset_config}")
    print(f"Task types: {args.task_types}")
    
    all_sft_samples = []
    
    for task_type in args.task_types:
        print(f"\nProcessing {task_type} samples...")
        samples = load_samples_from_yaml(args.dataset_config, task_type)
        print(f"Loaded {len(samples)} {task_type} samples")
        
        sft_samples = convert_to_sft_format(samples, task_type)
        print(f"Converted to {len(sft_samples)} SFT format samples")
        
        all_sft_samples.extend(sft_samples)
    
    print(f"\nTotal SFT samples: {len(all_sft_samples)}")
    
    # Save the dataset
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_sft_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to: {args.output_file}")
    
    # Print some statistics
    task_counts = {}
    for sample in all_sft_samples:
        # Determine task type from the user message
        user_message = sample["messages"][1]["content"]
        if "deficiency" in user_message.lower():
            task_type = "deficiency"
        elif "compare" in user_message.lower():
            task_type = "comparison"
        else:
            task_type = "score"
        
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print("\nDataset statistics:")
    for task_type, count in task_counts.items():
        print(f"  {task_type}: {count} samples")


if __name__ == "__main__":
    main()
