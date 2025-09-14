from typing import Tuple
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLVisionFlashAttention2,
    apply_rotary_pos_emb_flashatt,
    flash_attn_varlen_func,
)
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from PIL import Image
from torch.utils.data import Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.trainer import Qwen2VLGRPOTrainer, GRPOConfig
import yaml
import json
import random
import math
import torch
# Restoring imports for OpenAI API call
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables for the OpenAI API
load_dotenv()

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------


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


# ----------------------- Script Args (Multi-Task) -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """Script arguments for multi-task GRPO training/evaluation.
    
    NOTE: `image_root` is now configured inside the dataset YAML file, not here.
    """

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
    score_reward_threshold: Optional[float] = field(
        default=0.35, metadata={"help": "Threshold for score reward (abs diff). Default 0.35 for 1-5 scale"}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "YAML file path containing all dataset configurations"}
    )
    score_prompt_file: Optional[str] = field(
        default=None, metadata={"help": "Optional text file path that contains the evaluation prompt for scoring"}
    )
    deficiency_f1_threshold: Optional[float] = field(
        default=0.7, metadata={"help": "Threshold for deficiency category-level F1 to grant reward"}
    )


# Load prompts from files
def load_prompt_from_file(prompt_file: str, default_prompt: str = "") -> str:
    """Load prompt text from file, with fallback to default if file doesn't exist."""
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load prompt from {prompt_file}: {e}")
    return default_prompt

# Define prompt file paths
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
SYSTEM_PROMPT_FILE = os.path.join(PROMPTS_DIR, "system_prompt.txt")
SCORE_QUESTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "score_question_prompt.txt")
DEFICIENCY_PROMPT_FILE = os.path.join(PROMPTS_DIR, "deficiency_prompt.txt")
CLASSIFY_CATEGORIES_PROMPT_FILE = os.path.join(PROMPTS_DIR, "classify_categories_prompt.txt")
COMPARE_QUESTION_PROMPT_FILE = os.path.join(PROMPTS_DIR, "compare_question_prompt.txt")

# Load prompts from files
SYSTEM_PROMPT = load_prompt_from_file(SYSTEM_PROMPT_FILE)
SCORE_QUESTION_PROMPT = load_prompt_from_file(SCORE_QUESTION_PROMPT_FILE)
DEFICIENCY_PROMPT = load_prompt_from_file(DEFICIENCY_PROMPT_FILE)
CLASSIFY_CATEGORIES_PROMPT = load_prompt_from_file(CLASSIFY_CATEGORIES_PROMPT_FILE)
COMPARE_QUESTION_PROMPT = load_prompt_from_file(COMPARE_QUESTION_PROMPT_FILE)


class LazyMultiTaskDataset(Dataset):
    """Multi-task dataset that loads samples for scoring, deficiency detection, and comparison."""

    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args

        self.score_samples = []
        self.deficiency_samples = []
        self.comparison_samples = []

        dataset_config_path = getattr(script_args, "dataset_config", None)
        if dataset_config_path:
            print(f"Loading all datasets from unified config: {dataset_config_path}")
            self._load_from_unified_yaml(dataset_config_path)
        else:
            # Fallback to individual dataset paths for backward compatibility
            score_yaml_path = getattr(script_args, "dataset_score", None)
            if score_yaml_path:
                print(f"Loading score samples from {score_yaml_path}")
                self.score_samples = self._load_samples_from_yaml(score_yaml_path)

            deficiency_yaml_path = getattr(script_args, "dataset_deficiency", None)
            if deficiency_yaml_path:
                print(f"Loading deficiency samples from {deficiency_yaml_path}")
                self.deficiency_samples = self._load_samples_from_yaml(deficiency_yaml_path)

            comparison_yaml_path = getattr(script_args, "dataset_comparison", None)
            if comparison_yaml_path:
                print(f"Loading comparison samples from {comparison_yaml_path}")
                self.comparison_samples = self._load_samples_from_yaml(comparison_yaml_path)

        if not self.score_samples and not self.deficiency_samples and not self.comparison_samples:
            raise ValueError("Please provide at least one dataset: --dataset_config or individual dataset paths")

        self.total_len = len(self.score_samples) + len(self.deficiency_samples) + len(self.comparison_samples)

        prompt_file = getattr(script_args, "score_prompt_file", None)
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as pf:
                self.score_prompt_text = pf.read().strip()
        else:
            self.score_prompt_text = SCORE_QUESTION_PROMPT

        self.deficiency_prompt_text = DEFICIENCY_PROMPT
        self.comparison_prompt_text = COMPARE_QUESTION_PROMPT

    def _load_from_unified_yaml(self, data_path: str):
        """Load samples from unified YAML config that contains all task types."""
        if not data_path or not os.path.exists(data_path):
            raise ValueError(f"Dataset config file not found: {data_path}")
        
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}, must be a .yaml file.")
        
        with open(data_path, "r") as f:
            cfg = yaml.safe_load(f)
            
            # Load score samples
            if "score" in cfg:
                print(f"Loading score samples from unified config")
                self.score_samples = self._load_task_samples(cfg["score"])
            
            # Load deficiency samples  
            if "deficiency" in cfg:
                print(f"Loading deficiency samples from unified config")
                self.deficiency_samples = self._load_task_samples(cfg["deficiency"])
            
            # Load comparison samples
            if "compare" in cfg:
                print(f"Loading comparison samples from unified config")
                self.comparison_samples = self._load_task_samples(cfg["compare"])

    def _load_task_samples(self, task_config):
        """Load samples for a specific task type from the unified config."""
        samples = []
        
        if not isinstance(task_config, list):
            task_config = [task_config]
            
        for ds in task_config:
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

    def _load_samples_from_yaml(self, data_path: str):
        samples = []
        if not data_path.endswith(".yaml"):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r") as f:
            cfg = yaml.safe_load(f)
            for ds in cfg.get("datasets", []):
                path = ds.get("json_path")
                strategy = ds.get("sampling_strategy", "all")
                image_root = ds.get("image_root")

                if path.endswith(".jsonl"):
                    data_list = [json.loads(line) for line in open(path, "r")]
                elif path.endswith(".json"):
                    data_list = json.load(open(path, "r"))
                else:
                    raise ValueError(f"Unsupported file type: {path}")

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

                for sample in data_list:
                    sample['image_root'] = image_root

                samples.extend(data_list)
        return samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < len(self.score_samples):
            task_type = "score"
            example = self.score_samples[index]
        elif index < len(self.score_samples) + len(self.deficiency_samples):
            task_type = "deficiency"
            deficiency_index = index - len(self.score_samples)
            example = self.deficiency_samples[deficiency_index]
        else:
            task_type = "comparison"
            comparison_index = index - len(self.score_samples) - len(self.deficiency_samples)
            example = self.comparison_samples[comparison_index]

        sample = {"task": task_type}

        if task_type == "score":
            sample["prompt_text"] = self.score_prompt_text
            sol = example.get("score", None) or example.get("gt_score_norm", None)
            # When score is a dict with multiple dimensions, use only the overall value
            if isinstance(sol, dict):
                sol = sol.get("overall")
            sample["solution"] = sol
            sample["score_reward_threshold"] = self.script_args.score_reward_threshold
        elif task_type == "deficiency":
            sample["prompt_text"] = self.deficiency_prompt_text
            sample["solution"] = example.get("deficiencies", [])
            sample["deficiency_f1_threshold"] = self.script_args.deficiency_f1_threshold
        else:  # comparison task
            sample["prompt_text"] = self.comparison_prompt_text
            sample["solution"] = example.get("result")
            sample["system_prompt"] = SYSTEM_PROMPT
            sample["custom_question"] = self.comparison_prompt_text

        if task_type == "comparison":
            # Handle comparison task with multiple images
            image_root = example.get("image_root")
            
            # Reference image
            ref_rel = example.get("ref_image")
            ref_fp = os.path.join(image_root, ref_rel) if image_root else ref_rel
            if not os.path.exists(ref_fp):
                raise FileNotFoundError(f"Reference image not found: {ref_fp}")
            sample["ref_image"] = Image.open(ref_fp).convert("RGB")
            sample["ref_image_path"] = ref_fp
            
            # Slide A
            A_rel = example.get("ImageA")
            A_fp = os.path.join(image_root, A_rel) if image_root else A_rel
            if not os.path.exists(A_fp):
                raise FileNotFoundError(f"Slide A not found: {A_fp}")
            sample["imageA"] = Image.open(A_fp).convert("RGB")
            sample["imageA_path"] = A_fp
            
            # Slide B
            B_rel = example.get("ImageB")
            B_fp = os.path.join(image_root, B_rel) if image_root else B_rel
            if not os.path.exists(B_fp):
                raise FileNotFoundError(f"Slide B not found: {B_fp}")
            sample["imageB"] = Image.open(B_fp).convert("RGB")
            sample["imageB_path"] = B_fp

            # For comparison tasks, we don't use the standard prompt format
            # The trainer will handle the special format
            return sample
        else:
            # Handle single image tasks (score and deficiency)
            image_root = example.get("image_root")
            image_rel = example.get("image") or example.get("image_path")
            if image_rel is None:
                raise KeyError("Neither 'image' nor 'image_path' found in sample")

            image_path = os.path.join(image_root, image_rel) if image_root else image_rel

            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, trying another random sample of the same type")
                if task_type == "score":
                    new_index = random.randint(0, len(self.score_samples) - 1)
                    example = self.score_samples[new_index]
                else:  # deficiency
                    new_index = random.randint(0, len(self.deficiency_samples) - 1)
                    example = self.deficiency_samples[new_index]

                image_root = example.get("image_root")
                next_rel = example.get("image") or example.get("image_path", "")
                image_path = os.path.join(image_root, next_rel) if image_root else next_rel

            image = Image.open(image_path).convert("RGB")
            sample["image"] = image
            sample["image_path"] = image_path

            sample["prompt"] = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": sample["prompt_text"]},
                    ],
                },
            ]

            return sample


# --- ADDED: Pydantic model for structured LLM output ---
class DeficiencyCategoryBooleans(BaseModel):
    composition_layout: bool
    typography: bool
    imagery_visualizations: bool


# --- MODIFIED: Deficiency categories defined globally for reuse ---
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

# --- ADDED: Reverse mapping from specific deficiency to its parent category ---
DEFICIENCY_TO_CATEGORY = {
    deficiency: category
    for category, deficiencies in DEFICIENCY_CATEGORIES.items()
    for deficiency in deficiencies
}


# --- MODIFIED: `classify_deficiencies` function reverted to use an LLM for classification ---
def classify_deficiencies(model_output_text: str) -> List[str]:
    """
    Uses an external LLM (OpenAI API) to classify the model's free-text output
    into a predefined set of main deficiency categories.
    """
    # Extract content specifically from the <answer> tag for analysis.
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    match_answer = re.search(answer_tag_pattern, model_output_text, re.DOTALL)

    if match_answer:
        text_to_analyze = match_answer.group(1).strip()
    else:
        # Fallback to the original text if no <answer> tag is found.
        text_to_analyze = model_output_text.strip()

    # If there's no text to analyze, or it's a "no deficiencies" case, return an empty list.
    if not text_to_analyze or "no deficiencies" in text_to_analyze.lower():
        return []

    # The four main categories for the LLM to classify against.
    main_categories = list(DEFICIENCY_CATEGORIES.keys())

    prompt = CLASSIFY_CATEGORIES_PROMPT.format(
        categories=json.dumps(main_categories, indent=2),
        input_text=text_to_analyze
    )

    for attempt in range(3):
        try:
            client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE_URL")
            )

            model_type = os.getenv("MODEL_TYPE", "gpt-4o-2024-08-06")

            completion = client.chat.completions.parse(
                model=model_type,
                messages=[
                    {"role": "system", "content": "You are an expert in slide design analysis. Classify the user's text into the provided categories accurately."},
                    {"role": "user", "content": prompt}
                ],
                response_format=DeficiencyCategoryBooleans,
                temperature=0.0
            )

            result = completion.choices[0].message.parsed
            if not result:
                return []

            category_bools = {
                "Composition & Layout": getattr(result, "composition_layout", False),
                "Typography": getattr(result, "typography", False),
                "Imagery & Visualizations": getattr(result, "imagery_visualizations", False),
            }

            valid_categories = [name for name, is_present in category_bools.items() if is_present]
            return valid_categories

        except Exception as e:
            print(f"\033[31mError calling OpenAI API (attempt {attempt + 1}/3): {e}\033[0m")
            if attempt == 2:  # Last attempt
                return []
    return []


def verify_deficiency(completion_content, ground_truth_deficiencies, f1_threshold: float = 0.7, **kwargs):
    """
    Verifies the model's output based on the F1 score of deficiency CATEGORIES.
    
    The reward is 1.0 if the category-level F1 score is > 0.7, otherwise 0.0.
    This version first extracts the answer from within <answer>...</answer> tags.
    If no answer tags are found, returns 0.0 reward.
    """
    # --- MODIFICATION START: Extract content from <answer> tags ---
    # Use regex to find the content within <answer>...</answer>
    # re.DOTALL allows '.' to match newlines, in case the answer spans multiple lines.
    match = re.search(r"<answer>(.*?)</answer>", completion_content, re.DOTALL)

    if match:
        # If tags are found, use the content within them.
        # .strip() removes any leading/trailing whitespace.
        answer_content = match.group(1).strip()
    else:
        # If no tags are found, return 0.0 reward (no API call)
        return 0.0
    # --- MODIFICATION END ---

    # Get ground truth categories directly from the solution data
    gt_categories = {
        item["category"] for item in ground_truth_deficiencies 
        if "category" in item and item["category"] is not None
    }

    if not gt_categories:
        return 0.0

    # Get predicted categories from the model's extracted answer text via the LLM classifier.
    predicted_categories = set(classify_deficiencies(answer_content))

    # --- Handle edge cases before calculating F1 score ---
    if not gt_categories:
        return 1.0 if not predicted_categories else 0.0

    if not predicted_categories:
        # If GT has deficiencies but the model predicted none, the reward is 0.
        return 0.0

    # --- Calculate Precision, Recall, and F1 Score at the category level ---
    true_positives = len(gt_categories.intersection(predicted_categories))

    # Add a small epsilon to avoid division by zero
    precision = true_positives / len(predicted_categories)
    recall = true_positives / len(gt_categories)

    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    # --- Determine the final reward based on the F1 score threshold ---
    return 1.0 if f1_score > f1_threshold else 0.0


def calculate_task_metrics(content, true_sol, task_type, answer_content, **kwargs):
    """
    Calculate detailed metrics for each task type (F1, accuracy, etc.)
    Note: This function assumes answer_content is already extracted from <answer> tags.
    If no answer tags were found, this function should not be called.
    """
    metrics = {}
    
    if task_type == 'score':
        # For score task, calculate MAE and accuracy
        score_match = re.search(r'(\d+\.?\d*)', answer_content)
        if score_match:
            model_score = float(score_match.group(1))
            mae = abs(model_score - true_sol)
            metrics['mae'] = mae
            metrics['model_score'] = model_score
            metrics['gt_score'] = true_sol
        else:
            metrics['mae'] = float('inf')
            metrics['model_score'] = None
            metrics['gt_score'] = true_sol
            
    elif task_type == 'deficiency':
        # For deficiency task, calculate F1, precision, recall
        predicted_categories = set(classify_deficiencies(answer_content))
        gt_categories = {
            item.get("category")
            for item in true_sol
            if item.get("category") is not None
        }
        
        if not gt_categories and not predicted_categories:
            f1_score = 1.0
            precision = 1.0
            recall = 1.0
        elif not gt_categories or not predicted_categories:
            f1_score = 0.0
            precision = 0.0
            recall = 0.0
        else:
            tp = len(gt_categories.intersection(predicted_categories))
            precision = tp / len(predicted_categories) if len(predicted_categories) > 0 else 0.0
            recall = tp / len(gt_categories) if len(gt_categories) > 0 else 0.0
            f1_score = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
        
        metrics['f1'] = f1_score
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['predicted_categories'] = list(predicted_categories)
        metrics['gt_categories'] = list(gt_categories)
        
    elif task_type == 'comparison':
        # For comparison task, calculate accuracy
        answer_text = answer_content.strip()
        is_correct = answer_text == true_sol
        metrics['accuracy'] = 1.0 if is_correct else 0.0
        metrics['predicted'] = answer_text
        metrics['gt'] = true_sol
    
    return metrics


def accuracy_reward(completions, solution, task, image_path=None, score_reward_threshold=None, **kwargs):
    """
    Dispatcher reward function. Calls the appropriate reward logic based on the task.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    answer_tag_pattern = r"<answer>(.*?)</answer>"

    num_gen = len(solution) // len(contents) if len(contents) > 0 else 1
    subsampled_solutions = solution[::max(1, num_gen)]
    subsampled_tasks = task[::max(1, num_gen)]

    thresholds = score_reward_threshold
    subsampled_thresholds = thresholds[::max(1, num_gen)] if isinstance(thresholds, (list, tuple)) else [thresholds] * len(subsampled_solutions)
    if not any(isinstance(t, float) for t in subsampled_thresholds):
        subsampled_thresholds = [0.35] * len(subsampled_solutions)

    # Deficiency F1 thresholds (may arrive via batch kwargs)
    def_f1_thresholds_in = kwargs.get("deficiency_f1_threshold")
    subsampled_def_f1_thresholds = (
        def_f1_thresholds_in[::max(1, num_gen)] if isinstance(def_f1_thresholds_in, (list, tuple)) else [def_f1_thresholds_in] * len(subsampled_solutions)
    ) if def_f1_thresholds_in is not None else [0.7] * len(subsampled_solutions)

    # Track rewards by task type for logging
    task_rewards = {"score": [], "deficiency": [], "comparison": []}
    # Track metrics for wandb logging
    task_metrics = {"score": [], "deficiency": [], "comparison": []}

    for i, (content, true_sol, task_type) in enumerate(zip(contents, subsampled_solutions, subsampled_tasks)):
        reward = 0.0
        metrics = {}
        try:
            match_answer = re.search(answer_tag_pattern, content, re.DOTALL)
            if match_answer:
                answer_content = match_answer.group(1).strip()
                
                # Calculate detailed metrics for this sample
                metrics = calculate_task_metrics(content, true_sol, task_type, answer_content, **kwargs)

                if task_type == 'score':
                    score_match = re.search(r'(\d+\.?\d*)', answer_content)
                    if score_match:
                        model_score = float(score_match.group(1))
                        threshold_value = subsampled_thresholds[i] if i < len(subsampled_thresholds) else 0.35
                        if abs(model_score - true_sol) < threshold_value:
                            reward = 1.0

                elif task_type == 'deficiency':
                    f1_thr = subsampled_def_f1_thresholds[i] if i < len(subsampled_def_f1_thresholds) else 0.7
                    reward = verify_deficiency(content, true_sol, f1_threshold=f1_thr)
                
                elif task_type == 'comparison':
                    # For comparison tasks, use the comparison reward function
                    comparison_rewards = comparison_reward([completion], [true_sol], [task_type], **kwargs)
                    reward = comparison_rewards[0] if comparison_rewards else 0.0
            else:
                # If no answer tags found, reward and metrics are 0
                reward = 0.0
                metrics = {}

        except Exception:
            reward = 0.0
            metrics = {}
        rewards.append(reward)
        task_rewards[task_type].append(reward)
        task_metrics[task_type].append(metrics)

    if os.getenv("DEBUG_MODE") == "true":
        try:
            current_rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH", "./debug_log_eval_score_rl.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Rank: {current_rank} -------------\n")
                for i, content in enumerate(contents):
                    f.write(f"Task: {subsampled_tasks[i]}\n")
                    # 如果image_path是列表，只取第一个；否则直接使用
                    image_path_to_log = image_path[0] if isinstance(image_path, list) and len(image_path) > 0 else image_path
                    f.write(f"Image Path: {image_path_to_log}\n")
                    f.write(f"Reward: {rewards[i]}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Ground Truth: {subsampled_solutions[i]}\n")

                    if subsampled_tasks[i] == 'deficiency':
                        try:
                            # Extract answer content
                            match_answer_dbg = re.search(answer_tag_pattern, content, re.DOTALL)
                            answer_content_dbg = match_answer_dbg.group(1).strip() if match_answer_dbg else content

                            # Predicted categories via LLM
                            predicted_categories = set(classify_deficiencies(answer_content_dbg))

                            # Ground truth categories directly from solution data
                            gt_categories = {
                                item.get("category")
                                for item in subsampled_solutions[i]
                                if item.get("category") is not None
                            }

                            # Compute F1 (same as verify_deficiency)
                            if not gt_categories and not predicted_categories:
                                f1_score_dbg = 1.0
                            elif not gt_categories or not predicted_categories:
                                f1_score_dbg = 0.0
                            else:
                                tp_dbg = len(gt_categories.intersection(predicted_categories))
                                precision_dbg = tp_dbg / len(predicted_categories) if len(predicted_categories) > 0 else 0.0
                                recall_dbg = tp_dbg / len(gt_categories) if len(gt_categories) > 0 else 0.0
                                f1_score_dbg = 0.0 if (precision_dbg + recall_dbg) == 0 else 2 * (precision_dbg * recall_dbg) / (precision_dbg + recall_dbg)

                            # Write detailed logs
                            f.write(f"Predicted Categories: {sorted(list(predicted_categories))}\n")
                            f.write(f"GT Categories: {sorted(list(gt_categories))}\n")
                            f.write(f"F1(Category-level): {f1_score_dbg:.4f}\n")
                        except Exception as e:
                            f.write(f"Failed deficiency detailed logging: {e}\n")

                    f.write(f"{'=' * 40}\n")
        except Exception:
            pass
    
    # Store metrics in a global variable for trainer to access
    import threading
    if not hasattr(threading.current_thread(), 'task_metrics'):
        threading.current_thread().task_metrics = {}
    threading.current_thread().task_metrics = task_metrics
    
    return rewards


def format_reward(completions, solution, task, **kwargs):
    """
    Checks for the exact <think>...</think><answer>...</answer> structure.
    
    A reward of 1.0 is given only if the output contains exactly one <think> block
    followed by exactly one <answer> block. For deficiency tasks with "No deficiencies"
    as the ground truth, it also validates that the answer content matches.
    """
    # This pattern requires the string to contain only the think/answer structure,
    # allowing for surrounding whitespace.
    think_answer_pattern = r"^\s*<think>.*?</think>\s*<answer>(.*?)</answer>\s*$"

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    # Subsample solution and task to match the number of completions
    num_gen = len(solution) // len(completion_contents) if len(completion_contents) > 0 else 1
    subsampled_solutions = solution[::max(1, num_gen)]
    subsampled_tasks = task[::max(1, num_gen)]

    for content, true_sol, task_type in zip(completion_contents, subsampled_solutions, subsampled_tasks):
        reward = 0.0

        # --- MODIFICATION START ---
        # First, perform a strict count to ensure exactly one of each tag exists.
        # This prevents rewarding outputs with multiple <think>/<answer> pairs.
        is_single_tag_pair = (
            content.count("<think>") == 1
            and content.count("</think>") == 1
            and content.count("<answer>") == 1
            and content.count("</answer>") == 1
        )

        if is_single_tag_pair:
            # If the counts are correct, now validate the overall structure with the regex.
            # re.DOTALL ensures '.' matches newline characters within the tags.
            match = re.fullmatch(think_answer_pattern, content.strip(), re.DOTALL)

            if match:
                # Structure is correct, now check for the special "No deficiencies" case
                if task_type == 'deficiency' and not true_sol:
                    # Ground truth expects "No deficiencies"
                    answer_content = match.group(1).strip()
                    # Clean the answer for robust comparison
                    cleaned_answer = re.sub(r'[\s\W_]+', '', answer_content).lower()
                    if cleaned_answer == "nodeficiencies":
                        reward = 1.0
                    # else reward remains 0.0
                else:
                    # For all other cases, the correct structure is sufficient for a reward of 1.0
                    reward = 1.0
        # If tag counts are incorrect, reward remains 0.0
        # --- MODIFICATION END ---

        rewards.append(reward)

    return rewards


def comparison_reward(completions, solution, task, **kwargs):
    """
    For comparison tasks only:
      - Extract text from the <answer> tag.
      - If it exactly matches the solution (e.g., "Slide A" or "Slide B"), assign a reward of 1.0; otherwise, 0.0.
      - Preserve DEBUG logs by writing each match result to a file.
    """
    contents = [c[0]["content"] for c in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'

    # Subsample solution and task to match the number of completions
    num_gen = len(solution) // len(contents) if len(contents) > 0 else 1
    subsampled_solutions = solution[::max(1, num_gen)]
    subsampled_tasks = task[::max(1, num_gen)]

    for idx, (content, true_sol, task_type) in enumerate(zip(contents, subsampled_solutions, subsampled_tasks)):
        reward = 0.0
        answer_text = ""
        
        if task_type == 'comparison':
            try:
                m = re.search(answer_tag_pattern, content, re.DOTALL)
                if m:
                    answer_text = m.group(1).strip()
                    pat = re.compile(rf"^{re.escape(true_sol)}$")
                    if pat.fullmatch(answer_text):
                        reward = 1.0
            except Exception as e:
                print(f"Error in computing comparison reward at idx {idx}:", e)
        else:
            # For non-comparison tasks, return 0 reward
            reward = 0.0

        rewards.append(reward)

        # DEBUG logging
        if os.getenv("DEBUG_MODE") == "true" and task_type == 'comparison':
            try:
                current_rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                log_path = os.getenv("LOG_PATH", "./debug_log_comparison.txt")
                
                # Extract image paths from kwargs
                ref_image_path = kwargs.get('ref_image_path', [''])[idx] if 'ref_image_path' in kwargs else ''
                imageA_path = kwargs.get('imageA_path', [''])[idx] if 'imageA_path' in kwargs else ''
                imageB_path = kwargs.get('imageB_path', [''])[idx] if 'imageB_path' in kwargs else ''
                
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"----- {current_time} Rank:{current_rank} Index:{idx} -----\n")
                    f.write(f"Expected: {true_sol!r}\n")
                    f.write(f"Answer:   {answer_text!r}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Reward:   {reward}\n")
                    f.write(f"Ref Image Path: {ref_image_path}\n")
                    f.write(f"Image A Path: {imageA_path}\n")
                    f.write(f"Image B Path: {imageB_path}\n\n")
            except Exception:
                pass

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "comparison": comparison_reward,
}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    dataset = LazyMultiTaskDataset(script_args)

    # Use the unified trainer that handles all task types
    trainer_cls = Qwen2VLGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
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
