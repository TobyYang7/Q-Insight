# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import os

device = "cuda:0"
seed = 42
set_seed(seed)
MODEL_PATH = "ByteDance/Q-Insight"
SUBFOLDER = "score_degradation"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
    subfolder=SUBFOLDER
)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, subfolder=SUBFOLDER)
template = "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

image_path = "./test_figs/img_score.jpg"
SCORE_QUESTION_PROMPT = 'What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer in JSON format with the following keys: "rating": The score.'


message = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": SCORE_QUESTION_PROMPT
            },
            {"type": "image", "image": f"file://{image_path}"}
        ]
    }
]

text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
image_inputs, video_inputs = process_vision_info([message])
inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

gen_config = GenerationConfig(
  do_sample=True, 
  temperature=1.0,
  top_k=50,     
  top_p=0.95,
  max_new_tokens=1024,
)

generated_ids = model.generate(
  **inputs,
  generation_config=gen_config,
  use_cache=True,
)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("Model Response:")
print(output_text)
