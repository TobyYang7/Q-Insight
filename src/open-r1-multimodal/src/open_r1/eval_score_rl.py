import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

import yaml
import json
import random
import math
import torch

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
from typing import Tuple


def _custom_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    if position_embeddings is None:
        # Avoid dtype issues
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos().float()
        sin = emb.sin().float()
    else:
        cos, sin = position_embeddings
        cos = cos.to(torch.float)
        sin = sin.to(torch.float)
    q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
    ).reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


Qwen2_5_VLVisionFlashAttention2.forward = _custom_flash_attn_forward


# ----------------------- Script Args (Score-only) -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for score-only GRPO training/evaluation."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056, metadata={"help": "Maximum number of pixels for the image"}
    )
    min_pixels: Optional[int] = field(
        default=3136, metadata={"help": "Minimum number of pixels for the image"}
    )
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Root directory of the image"}
    )
    score_reward_threshold: Optional[float] = field(
        default=0.35, metadata={"help": "Threshold for score reward (abs diff). Default 0.35 for 1-5 scale"}
    )
    dataset_score: Optional[str] = field(
        default=None, metadata={"help": "YAML file path for the quality scoring dataset"}
    )
    score_prompt_file: Optional[str] = field(
        default=None, metadata={"help": "Optional text file path that contains the evaluation prompt for scoring"}
    )


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

SCORE_QUESTION_PROMPT = (
    'What is your overall rating on the quality of this slide?'
    'The rating should be a float between 1 and 10, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality.'
    'You need to provide your detailed reasoning process.'
)


class LazyScoreDataset(Dataset):
    """Score-only dataset that loads samples from YAML -> JSON and builds prompts."""

    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args

        yaml_path = getattr(script_args, "dataset_score", None)
        if not yaml_path:
            raise ValueError("Please provide the dataset file: --dataset_score <path_to_yaml>")
        self.samples = self._load_samples_from_yaml(yaml_path)
        if not self.samples:
            raise ValueError("No samples loaded; please check your dataset file content and path.")
        self.total_len = len(self.samples)

        # Load prompt text
        # prompt_file = getattr(script_args, "score_prompt_file", None)
        # if prompt_file and os.path.exists(prompt_file):
        #     with open(prompt_file, "r", encoding="utf-8") as pf:
        #         self.prompt_text = pf.read().strip()
        # else:
        #     self.prompt_text = SCORE_QUESTION_PROMPT
        
        # fix
        self.prompt_text = SCORE_QUESTION_PROMPT

    def _load_samples_from_yaml(self, data_path: str):
        samples = []
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r") as f:
            cfg = yaml.safe_load(f)
            for ds in cfg.get("datasets", []):
                path = ds.get("json_path")
                strategy = ds.get("sampling_strategy", "all")
                # load JSON or JSONL
                if path.endswith(".jsonl"):
                    data_list = [json.loads(line) for line in open(path, "r")]
                elif path.endswith(".json"):
                    data_list = json.load(open(path, "r"))
                else:
                    raise ValueError(f"Unsupported file type: {path}")
                # sampling
                count = None
                if ":" in strategy:
                    strat, num_s = strategy.split(":")
                    if "%" in num_s:
                        count = math.ceil(int(num_s.rstrip("%")) * len(data_list) / 100)
                    else:
                        count = int(num_s)
                    strategy = strat
                if strategy == "first" and count is not None:
                    data_list = data_list[:count]
                elif strategy == "end" and count is not None:
                    data_list = data_list[-count:]
                elif strategy == "random" and count is not None:
                    random.shuffle(data_list)
                    data_list = data_list[:count]
                # print(f"Loaded {len(data_list)} samples from {path}")
                samples.extend(data_list)
        return samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        example = self.samples[index]
        sample = {}

        # Ground truth numeric score
        # Prefer new dataset key 'score' (1-10), fallback to 'gt_score_norm' (1-5)
        sample["task"] = "score"
        sol = example.get("score", None)
        if sol is None:
            sol = example.get("gt_score_norm", None)
        sample["solution"] = sol
        sample["score_reward_threshold"] = self.script_args.score_reward_threshold

        # Image loading
        image_root = self.script_args.image_root
        # Support both 'image' and 'image_path'
        image_rel = example.get("image") if "image" in example else example.get("image_path")
        if image_rel is None:
            raise KeyError("Neither 'image' nor 'image_path' found in sample")
        if image_root is None:
            image_path = image_rel
        else:
            image_path = os.path.join(image_root, image_rel)
        if True:
            # Fallback: try another sample if path missing
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, trying another sample")
                new_index = random.randint(0, len(self.samples) - 1)
                example = self.samples[new_index]
                next_rel = example.get("image") if "image" in example else example.get("image_path", "")
                image_path = os.path.join(image_root, next_rel) if image_root else next_rel
            image = Image.open(image_path).convert("RGB")

        sample["image"] = image
        sample["image_path"] = image_path

        # Prompt
        sample["prompt"] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt_text},
                ],
            },
        ]

        return sample


def score_reward(completions, solution, task=None, image_path=None, score_reward_threshold=None, **kwargs):
    """Compute reward for score task only.

    - Extract JSON inside <answer>...</answer> and read "rating".
    - Reward 1.0 if |model_score - gt_score_norm| < threshold, else 0.0.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    # Patterns
    answer_tag_pattern = r"<answer>(.*?)</answer>"

    # Align lengths with trainer's repetition strategy
    num_gen = len(solution) // len(contents) if len(contents) > 0 else 1
    subsampled_solutions = solution[::max(1, num_gen)]

    # Threshold from dataset field if available
    threshold = 0.35 if score_reward_threshold is None else score_reward_threshold

    for content, true_sol in zip(contents, subsampled_solutions):
        reward = 0.0
        try:
            # <answer>...</answer>
            match_answer = re.search(answer_tag_pattern, content, re.DOTALL)
            if match_answer:
                # Extract the score from the answer
                answer_content = match_answer.group(1).strip()
                
                # Try to extract numeric score from various formats
                # Pattern 1: Direct number like "4.25"
                score_match = re.search(r'(\d+\.?\d*)', answer_content)
                if score_match:
                    try:
                        model_score = float(score_match.group(1))
                        # Calculate reward based on threshold
                        score_diff = abs(model_score - true_sol)
                        # Handle threshold as either single value or list
                        threshold_value = threshold[0] if isinstance(threshold, (list, tuple)) else threshold
                        if score_diff < threshold_value:
                            reward = 1.0
                        else:
                            reward = 0.0
                    except ValueError:
                        reward = 0.0
                else:
                    reward = 0.0
        except Exception:
            reward = 0.0
        rewards.append(reward)

    # Optional debug logging
    if os.getenv("DEBUG_MODE") == "true":
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                current_rank = torch.distributed.get_rank()
            else:
                current_rank = 0
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH", "./debug_log_eval_score_rl.txt")
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Rank: {current_rank} -------------\n")
                for i, content in enumerate(contents):
                    # Extract and display think process
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    think_content = think_match.group(1).strip() if think_match else "Not found"
                    
                    # Extract and display answer
                    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    answer_content = answer_match.group(1).strip() if answer_match else "Not found"
                    
                    # Get ground truth
                    ground_truth = subsampled_solutions[i] if i < len(subsampled_solutions) else 'N/A'
                    
                    f.write(f"Think: {think_content}\n")
                    f.write(f"Answer: {answer_content}\n")
                    f.write(f"Ground Truth: {ground_truth}\n")
                    
                    # Debug reward calculation
                    if answer_match:
                        score_match = re.search(r'(\d+\.?\d*)', answer_content)
                        if score_match:
                            try:
                                model_score = float(score_match.group(1))
                                score_diff = abs(model_score - ground_truth)
                                f.write(f"Model Score: {model_score}, Diff: {score_diff}, Threshold: {threshold}, Reward: {rewards[i] if i < len(rewards) else 'N/A'}\n")
                            except ValueError:
                                f.write(f"Score parsing failed\n")
                        else:
                            f.write(f"No numeric score found\n")
                    f.write(f"{'=' * 40}\n")
                
                f.write(f"Threshold: {threshold}\n")
                f.write(f"All Rewards: {rewards}\n")
        except Exception:
            pass

    return rewards


def format_reward(completions, **kwargs):
    """Check output format strictly for this task:
    - Must have <think>...</think> followed by <answer>...</answer>
    - Inside <answer> must be a flat JSON with keys {"reason": string, "score": float}
    - score must be within [1, 5]
    """
    think_answer_pattern = (
        r"^<think>\s*\n"  # <think> tag
        r".*?\n"  # content of think (non-greedy) until a newline
        r"\s*</think>\s*\n"  # </think>
        r"<answer>\s*\n"  # <answer>
        r"(\{[^\{\}]*\})"  # capture flat JSON
        r"\s*\n"  # newline after JSON
        r"\s*</answer>\s*$"  # </answer>
    )

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        ok = 0.0
        m = re.fullmatch(think_answer_pattern, content, re.DOTALL | re.MULTILINE)
        if m:
            json_str = m.group(1)
            # very light-weight validation: require both keys and a numeric score in [1,5]
            has_reason = re.search(r'\"reason\"\s*:\s*\"', json_str) is not None
            m_score = re.search(r'\"score\"\s*:\s*([\d\.]+)', json_str)
            if has_reason and m_score:
                try:
                    sc = float(m_score.group(1))
                    if 1.0 <= sc <= 5.0:
                        ok = 1.0
                except Exception:
                    ok = 0.0
        rewards.append(ok)
    return rewards


reward_funcs_registry = {
    "accuracy": score_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load dataset (score-only)
    dataset = LazyScoreDataset(script_args)

    trainer_cls = Qwen2VLGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        # max_pixels=script_args.max_pixels,
        max_pixels=1920*28*28,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


