from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image
import base64
import io

# --- Helper function to process and encode the image ---


def image_to_base64_uri(image_path: str, max_width: int = 1024) -> str:
    """
    Loads an image, resizes it to a maximum width while preserving
    the aspect ratio, and encodes it as a Base64 data URI.
    """
    try:
        # Open the image using Pillow
        img = Image.open(image_path)

        # Determine the image format (PNG, JPEG, etc.)
        img_format = img.format if img.format else 'PNG'

        # Check if the image needs resizing
        if img.width > max_width:
            # Calculate the new height to maintain aspect ratio
            aspect_ratio = img.height / img.width
            new_height = int(max_width * aspect_ratio)

            # Resize the image using a high-quality filter
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Save the (potentially resized) image to an in-memory buffer
        buffered = io.BytesIO()
        img.save(buffered, format=img_format)

        # Encode the bytes in the buffer to a Base64 string
        img_byte = buffered.getvalue()
        base64_str = base64.b64encode(img_byte).decode('utf-8')

        # Format as a data URI
        return f"data:image/{img_format.lower()};base64,{base64_str}"

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# --- Main script configuration ---
device = "cuda:0"
seed = 42
set_seed(seed)
# MODEL_PATH = "model/evalutator_deficiency_ckpt_500_0825"
MODEL_PATH = "model/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# if 'qwen2.5' in MODEL_PATH.lower():
#     SYSTEM_PROMPT = "You are a helpful assistant."

processor = AutoProcessor.from_pretrained(MODEL_PATH)
template = "First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

image_path = "paper-slide-crawler/downloads/acl2023/2023.acl-long.202/slides/22.jpg"

# --- Prompts remain the same ---
SCORE_QUESTION_PROMPT = (
    'What is your overall rating on the quality of this slide?'
    'The rating should be a float between 1 and 10, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality.'
    'You need to provide your detailed reasoning process.'
)
DEFICIENCY_PROMPT = (
    "Please provide a professional design critique of the accompanying slide. "
    "If there are no deficiencies, you should say 'No deficiencies'."
    "Otherwise, your analysis should identify any design deficiencies, explain the reasoning behind your critique, "
    "and offer specific, actionable suggestions for improvement. "
)

# --- Process the image and create the message payload ---
base64_image_uri = image_to_base64_uri(image_path)

if base64_image_uri:
    message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": DEFICIENCY_PROMPT
                },
                # Use the Base64 data URI instead of a file path
                {"type": "image", "image": base64_image_uri}
            ]
        }
    ]

    # --- Model inference (unchanged) ---
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
    print("\033[94mImage Path:\033[0m", image_path)  # 蓝色
    print("\033[92mModel Response:\033[0m")  # 绿色
    print("\033[93m" + output_text + "\033[0m")  # 黄色
else:
    print(f"Could not process the image at: {image_path}")
