import os
import json
import yaml
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    Qwen2VLProcessor,
    Qwen2ForConditionalGeneration,
)

# Reuse prompts from the other script if they exist
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPTS_DIR, "system_prompt.txt")
DEFICIENCY_PROMPT_FILE = os.path.join(PROMPTS_DIR, "deficiency_prompt.txt")
SCORE_QUESTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "score_question_prompt.txt")
COMPARE_QUESTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "compare_question_prompt.txt")


def load_prompt_from_file(prompt_file: str, default_prompt: str = "") -> str:
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return default_prompt


SYSTEM_PROMPT = load_prompt_from_file(SYSTEM_PROMPT_FILE, "You are a helpful assistant specialized in slide design analysis.")
DEFICIENCY_PROMPT = load_prompt_from_file(DEFICIENCY_PROMPT_FILE, "Please identify any design deficiencies in this slide.")
SCORE_QUESTION_PROMPT = load_prompt_from_file(SCORE_QUESTION_PROMPT_FILE, "Please rate the quality of this slide on a scale of 1 to 5.")
COMPARE_QUESTION_PROMPT = load_prompt_from_file(COMPARE_QUESTION_PROMPT_FILE, "Compare Slide A and Slide B based on the reference slide. Which one is better?")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Override the default `torch.dtype` and load the model under this dtype."})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Attention implementation to use (e.g., 'flash_attention_2')."})


@dataclass
class DataArguments:
    dataset_config: Optional[str] = field(default=None, metadata={"help": "YAML file containing all dataset configurations."})


@dataclass
class SFTTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output_sft_multi")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    bf16: bool = field(default=True)
    report_to: Optional[str] = field(default="wandb")
    run_name: Optional[str] = field(default="qwen-vl-sft-multi-task")


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


class MultiTaskSFTDataset(Dataset):
    def __init__(self, data_args: DataArguments, processor: Qwen2VLProcessor):
        self.processor = processor
        
        # Load from unified config if available
        if data_args.dataset_config:
            self.score_samples = load_samples_from_yaml(data_args.dataset_config, "score")
            self.deficiency_samples = load_samples_from_yaml(data_args.dataset_config, "deficiency")
            self.comparison_samples = load_samples_from_yaml(data_args.dataset_config, "compare")
        else:
            # Fallback to individual dataset configs for backward compatibility
            self.score_samples = load_samples_from_yaml(getattr(data_args, "dataset_score", None))
            self.deficiency_samples = load_samples_from_yaml(getattr(data_args, "dataset_deficiency", None))
            self.comparison_samples = load_samples_from_yaml(getattr(data_args, "dataset_comparison", None))
        
        self.total_len = len(self.score_samples) + len(self.deficiency_samples) + len(self.comparison_samples)
        if self.total_len == 0:
            raise ValueError("No samples loaded. Please provide at least one valid dataset config.")

    def __len__(self):
        return self.total_len

    def _format_deficiency_solution(self, deficiencies: List[Dict]) -> str:
        if not deficiencies:
            return "<think>I have analyzed the slide and found no design deficiencies.</think><answer>No deficiencies found.</answer>"

        thought = f"I have analyzed the slide and found {len(deficiencies)} deficiency/deficiencies. I will list them now."
        answer_parts = [f"- **{d['category']}**: {d['deficiency']}" for d in deficiencies]
        answer = "\n".join(answer_parts)
        return f"<think>{thought}</think><answer>\n{answer}\n</answer>"

    def _format_score_solution(self, score: Any) -> str:
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

    def _format_comparison_solution(self, result: str) -> str:
        thought = f"I have compared Slide A and Slide B against the reference. Based on my analysis, {result} is better."
        answer = result
        return f"<think>{thought}</think><answer>{answer}</answer>"

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < len(self.score_samples):
            task_type = "score"
            example = self.score_samples[index]
        elif index < len(self.score_samples) + len(self.deficiency_samples):
            task_type = "deficiency"
            example = self.deficiency_samples[index - len(self.score_samples)]
        else:
            task_type = "comparison"
            example = self.comparison_samples[index - len(self.score_samples) - len(self.deficiency_samples)]
        
        image_root = example.get("image_root", "")
        
        if task_type == "comparison":
            ref_img_path = os.path.join(image_root, example["ref_image"])
            img_a_path = os.path.join(image_root, example["ImageA"])
            img_b_path = os.path.join(image_root, example["ImageB"])
            
            images = [Image.open(p).convert("RGB") for p in [ref_img_path, img_a_path, img_b_path]]
            
            solution_text = self._format_comparison_solution(example["result"])
            
            # For comparison, the prompt includes placeholders for the three images.
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}, # Reference
                        {"type": "image"}, # Image A
                        {"type": "image"}, # Image B
                        {"type": "text", "text": COMPARE_QUESTION_PROMPT},
                    ],
                },
                {"role": "assistant", "content": solution_text}
            ]

        else: # Score or Deficiency
            img_path = os.path.join(image_root, example.get("image") or example.get("image_path"))
            images = [Image.open(img_path).convert("RGB")]
            
            if task_type == "score":
                prompt_text = SCORE_QUESTION_PROMPT
                solution = example.get("score") or example.get("gt_score_norm")
                solution_text = self._format_score_solution(solution)
            else: # deficiency
                prompt_text = DEFICIENCY_PROMPT
                solution_text = self._format_deficiency_solution(example.get("deficiencies", []))
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {"role": "assistant", "content": solution_text}
            ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        model_inputs = self.processor(text=[text], images=images, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in model_inputs.items()}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    model_dtype = getattr(torch, model_args.torch_dtype)
    model = Qwen2ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=True
    )
    
    train_dataset = MultiTaskSFTDataset(data_args, processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
