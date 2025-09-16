#!/usr/bin/env python3
"""
Fix script for score evaluation JSON files.
This script identifies and fixes abnormal predicted scores that are likely parsing errors.
"""

import json
import re
import argparse
import os
from typing import Dict, Any, List, Tuple


def extract_score_from_text(text: str) -> Tuple[float, bool]:
    """
    Extract score from model output text.
    Returns (score, is_valid) tuple.
    """
    # First try to extract from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        answer_content = text
    
    # Look for numeric patterns in the answer content
    # Try to find a reasonable score (between 0 and 10)
    score_patterns = [
        r'(\d+\.?\d*)',  # Any number
        r'(\d+\.\d+)',   # Decimal number
        r'(\d+)',        # Integer
    ]
    
    for pattern in score_patterns:
        matches = re.findall(pattern, answer_content)
        for match in matches:
            try:
                score = float(match)
                # Check if score is reasonable (between 0 and 10)
                if 0 <= score <= 10:
                    return score, True
            except ValueError:
                continue
    
    # If no reasonable score found, return None
    return None, False


def is_abnormal_score(predicted_score: float, ground_truth_score: float) -> bool:
    """
    Check if a predicted score is abnormal based on various criteria.
    """
    # Check for extremely large values (likely parsing errors)
    if predicted_score > 100:
        return True
    
    # Check for extremely large differences
    if abs(predicted_score - ground_truth_score) > 50:
        return True
    
    # Check for negative scores (should be between 0-10)
    if predicted_score < 0:
        return True
    
    return False


def fix_score_json(input_file: str, output_file: str = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Fix abnormal scores in a score evaluation JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input)
        dry_run: If True, only analyze without making changes
    
    Returns:
        Dictionary with fix statistics
    """
    if output_file is None:
        output_file = input_file
    
    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total_samples': 0,
        'abnormal_scores': 0,
        'fixed_scores': 0,
        'failed_extractions': 0,
        'abnormal_samples': []
    }
    
    if 'detailed_results' not in data:
        print("Error: No 'detailed_results' found in JSON file")
        return stats
    
    detailed_results = data['detailed_results']
    stats['total_samples'] = len(detailed_results)
    
    print(f"Processing {stats['total_samples']} samples...")
    
    # Process each sample
    for i, result in enumerate(detailed_results):
        if result.get('task') != 'score':
            continue
            
        predicted_score = result.get('predicted_score')
        ground_truth_score = result.get('ground_truth_score')
        model_output = result.get('model_output', '')
        
        if predicted_score is None or ground_truth_score is None:
            continue
        
        # Check if score is abnormal
        if is_abnormal_score(predicted_score, ground_truth_score):
            stats['abnormal_scores'] += 1
            stats['abnormal_samples'].append({
                'index': i,
                'slide_id': result.get('slide_id', 'unknown'),
                'original_score': predicted_score,
                'ground_truth': ground_truth_score,
                'mae': abs(predicted_score - ground_truth_score)
            })
            
            print(f"Found abnormal score: {predicted_score} (GT: {ground_truth_score}) in sample {i}")
            
            # Try to re-extract score from model output
            new_score, is_valid = extract_score_from_text(model_output)
            
            if is_valid and not is_abnormal_score(new_score, ground_truth_score):
                print(f"  -> Fixed to: {new_score}")
                if not dry_run:
                    result['predicted_score'] = new_score
                    # Recalculate metrics
                    mae = abs(new_score - ground_truth_score)
                    mse = (new_score - ground_truth_score) ** 2
                    threshold = 0.35
                    within_threshold = 1.0 if mae < threshold else 0.0
                    
                    result['metrics'] = {
                        'mae': mae,
                        'mse': mse,
                        'within_threshold': within_threshold,
                        'threshold': threshold
                    }
                stats['fixed_scores'] += 1
            else:
                print(f"  -> Could not extract valid score, marking as error")
                if not dry_run:
                    result['predicted_score'] = None
                    result['metrics'] = {
                        'mae': None,
                        'mse': None,
                        'within_threshold': None,
                        'threshold': 0.35,
                        'error': 'Failed to extract valid score'
                    }
                stats['failed_extractions'] += 1
    
    # Recalculate overall metrics if not dry run
    if not dry_run and stats['fixed_scores'] > 0 or stats['failed_extractions'] > 0:
        print("Recalculating overall metrics...")
        recalculate_overall_metrics(data)
    
    # Save the fixed file
    if not dry_run:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Fixed file saved to: {output_file}")
    
    return stats


def recalculate_overall_metrics(data: Dict[str, Any]) -> None:
    """
    Recalculate overall metrics after fixing scores.
    """
    detailed_results = data.get('detailed_results', [])
    
    # Filter valid score results
    valid_scores = []
    for result in detailed_results:
        if (result.get('task') == 'score' and 
            result.get('predicted_score') is not None and 
            result.get('ground_truth_score') is not None):
            valid_scores.append((result['predicted_score'], result['ground_truth_score']))
    
    if not valid_scores:
        print("Warning: No valid scores found for recalculation")
        return
    
    predicted_scores, gt_scores = zip(*valid_scores)
    
    # Calculate metrics
    mae = sum(abs(p - gt) for p, gt in valid_scores) / len(valid_scores)
    mse = sum((p - gt) ** 2 for p, gt in valid_scores) / len(valid_scores)
    rmse = mse ** 0.5
    
    # Threshold-based accuracy
    threshold = 0.35
    within_threshold = sum(1 for p, gt in valid_scores if abs(p - gt) < threshold) / len(valid_scores)
    
    # Update overall metrics
    if 'overall_metrics' not in data:
        data['overall_metrics'] = {}
    
    data['overall_metrics']['score'] = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'within_threshold': within_threshold,
        'threshold': threshold,
        'total_samples': len(detailed_results),
        'valid_predictions': len(valid_scores),
        'failed_predictions': len(detailed_results) - len(valid_scores)
    }
    
    print(f"Recalculated metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fix abnormal scores in evaluation JSON files")
    parser.add_argument("input_file", help="Input JSON file to fix")
    parser.add_argument("--output", "-o", help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    print(f"Processing file: {args.input_file}")
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
    
    stats = fix_score_json(args.input_file, args.output, args.dry_run)
    
    print(f"\n{'='*50}")
    print("FIX STATISTICS")
    print(f"{'='*50}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Abnormal scores found: {stats['abnormal_scores']}")
    print(f"Successfully fixed: {stats['fixed_scores']}")
    print(f"Failed extractions: {stats['failed_extractions']}")
    
    if stats['abnormal_samples'] and args.verbose:
        print(f"\nAbnormal samples:")
        for sample in stats['abnormal_samples'][:10]:  # Show first 10
            print(f"  Sample {sample['index']}: {sample['original_score']} -> GT: {sample['ground_truth']} (MAE: {sample['mae']:.2f})")
        if len(stats['abnormal_samples']) > 10:
            print(f"  ... and {len(stats['abnormal_samples']) - 10} more")
    
    return 0


if __name__ == "__main__":
    exit(main())

