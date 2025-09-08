#!/usr/bin/env python3
"""
Evaluation script for the slide quality scoring model, using a remote API for inference.
"""

import os
import json
import re
import argparse
from typing import Dict, List, Any
import base64
import io

# openai library is used to interact with the API
from openai import OpenAI
from PIL import Image
import numpy as np
from tqdm import tqdm

# System prompt and question prompt from training script
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

def resize_image_with_constraints(image: Image.Image) -> Image.Image:
    """
    Resize image following specific constraints to handle very large or small images.
    """
    MAX_LONG_SIDE = 1280
    MIN_SIDE = 28
    w, h = image.size

    # 1. Handle images that are too LARGE (downscale)
    if w > MAX_LONG_SIDE or h > MAX_LONG_SIDE:
        if w > h:
            # Width is the longest side
            new_w = MAX_LONG_SIDE
            new_h = int(h * (MAX_LONG_SIDE / w))
        else:
            # Height is the longest side
            new_h = MAX_LONG_SIDE
            new_w = int(w * (MAX_LONG_SIDE / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 2. Handle images that are too SMALL (upscale)
    elif w < MIN_SIDE or h < MIN_SIDE:
        if w < h:
            # Width is the shortest side
            new_w = MIN_SIDE
            new_h = int(h * (MIN_SIDE / w))
        else:
            # Height is the shortest side
            new_h = MIN_SIDE
            new_w = int(w * (MIN_SIDE / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def image_to_base64(image: Image.Image, format="JPEG") -> str:
    """Encodes a PIL image into a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def extract_score_from_response(response: str) -> float:
    """
    Extract numeric score from model response.
    """
    # Look for score in <answer> tags
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Try to extract numeric score
        score_match = re.search(r'(\d+\.?\d*)', answer_content)
        if score_match:
            try:
                return float(score_match.group(1))
            except ValueError:
                pass
    
    # Fallback: look for any number in the response
    numbers = re.findall(r'\b\d+\.?\d*\b', response)
    if numbers:
        try:
            return float(numbers[-1])  # Take the last number found
        except ValueError:
            pass
    
    return -1.0  # Return -1 if no valid score found


def load_test_data(json_path: str, image_root: str = None) -> List[Dict[str, Any]]:
    """
    Load test dataset from JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process image paths
    for item in data:
        if image_root:
            # Use image_root if provided
            image_rel = item.get('image', item.get('image_path', ''))
            item['full_image_path'] = os.path.join(image_root, image_rel)
        else:
            # Use absolute path or relative to JSON file location
            image_path = item.get('image_path', item.get('image', ''))
            if not os.path.isabs(image_path):
                json_dir = os.path.dirname(json_path)
                item['full_image_path'] = os.path.join(json_dir, image_path)
            else:
                item['full_image_path'] = image_path
        
        # Ensure absolute path
        if not os.path.isabs(item['full_image_path']):
            item['full_image_path'] = os.path.abspath(item['full_image_path'])
    
    return data


def evaluate_model(
    test_data_path: str,
    api_key: str,
    base_url: str,
    model_name: str,
    image_root: str = None,
    output_file: str = None,
) -> Dict[str, float]:
    """
    Evaluate the model on test dataset by calling a remote API.
    """
    # Initialize the API client instead of loading a local model
    print(f"Initializing API client for model: {model_name} at {base_url}")
    try:
        # The OpenAI client can automatically pick up the API key from the
        # OPENAI_API_KEY environment variable if the api_key argument is not provided.
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return {}

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = load_test_data(test_data_path, image_root)
    print(f"Loaded {len(test_data)} test samples")
    
    # Evaluation metrics
    predictions = []
    ground_truths = []
    results = []
    
    # Process each sample
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            # Load image
            image_path = sample['full_image_path']
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
                
            image = Image.open(image_path).convert('RGB')
            
            # **MODIFICATION**: Resize image before encoding
            image = resize_image_with_constraints(image)
            
            base64_image = image_to_base64(image)
            
            # Prepare payload for API call
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": SCORE_QUESTION_PROMPT}
                    ]
                }
            ]

            # Call the API
            response_payload = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            response = response_payload.choices[0].message.content
            
            # Extract score
            predicted_score = extract_score_from_response(response)
            ground_truth_score = sample['score']
            
            predictions.append(predicted_score)
            ground_truths.append(ground_truth_score)
            
            # Store detailed results, including the raw API response
            result = {
                'image_path': image_path,
                'ground_truth': ground_truth_score,
                'predicted': predicted_score,
                'response': response,
                'raw_response_payload': response_payload.model_dump(), # Save the full API response object
                'paper_id': sample.get('paper_id', ''),
                'conference': sample.get('conference', ''),
                'slide_id': sample.get('slide_id', '')
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {i} ({sample.get('full_image_path', 'N/A')}): {e}")
            continue
    
    # Calculate metrics
    valid_predictions = [(p, g) for p, g in zip(predictions, ground_truths) if p >= 0]
    
    if not valid_predictions:
        print("No valid predictions found!")
        return {}
    
    valid_pred, valid_gt = zip(*valid_predictions)
    valid_pred = np.array(valid_pred)
    valid_gt = np.array(valid_gt)
    
    # Calculate evaluation metrics
    mae = np.mean(np.abs(valid_pred - valid_gt))
    mse = np.mean((valid_pred - valid_gt) ** 2)
    rmse = np.sqrt(mse)
    
    # Accuracy within threshold (0.35 as used in training)
    threshold = 0.35
    accuracy = np.mean(np.abs(valid_pred - valid_gt) < threshold)
    
    # Correlation
    correlation = np.corrcoef(valid_pred, valid_gt)[0, 1] if len(valid_pred) > 1 else 0.0
    
    metrics = {
        'num_samples': len(test_data),
        'valid_predictions': len(valid_predictions),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'accuracy_0.35': accuracy,
        'correlation': correlation
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of test samples: {metrics['num_samples']}")
    print(f"Valid predictions: {metrics['valid_predictions']}")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f}")
    print(f"MSE (Mean Squared Error): {metrics['mse']:.4f}")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"Accuracy (within 0.35): {metrics['accuracy_0.35']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print("="*50)
    
    # Save detailed results if output file specified
    if output_file:
        output_data = {
            'metrics': metrics,
            'detailed_results': results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {output_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate slide quality scoring model via API')
    # Updated API-related arguments to be optional with defaults
    parser.add_argument('--api_key', type=str, default="None",
                       help='API key for the inference service. Can also be set via OPENAI_API_KEY env var.')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000/v1',
                       help='Base URL of the inference service')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the model to use for inference')

    parser.add_argument('--test_data', type=str,
                       default='/root/Q-Insight/paper-slide-crawler/dataset/slide_quality_test.json',
                       help='Path to test dataset JSON file')
    parser.add_argument('--image_root', type=str, 
                       default='/root/Q-Insight/paper-slide-crawler',
                       help='Root directory for images')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    output_file = os.path.join(args.output_dir, "results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    metrics = evaluate_model(
        test_data_path=args.test_data,
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        image_root=args.image_root,
        output_file=output_file,
    )

if __name__ == '__main__':
    main()
