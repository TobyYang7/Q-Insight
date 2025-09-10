#!/usr/bin/env python3
"""
数据集分割程序
将score_gpt.json和deficiency_gpt.json按9:1比例分割为训练集和测试集
确保两个任务使用相同的图片分割，保持一致性
"""

import json
import random
import os
from typing import List, Dict, Any, Tuple

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_dataset(data: List[Dict[str, Any]], train_ratio: float = 0.9, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    分割数据集为训练集和测试集
    
    Args:
        data: 要分割的数据列表
        train_ratio: 训练集比例，默认0.9
        random_seed: 随机种子，确保可重复性
    
    Returns:
        (train_data, test_data): 训练集和测试集
    """
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算分割点
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    
    # 分割数据
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]
    
    return train_data, test_data

def get_image_paths(data: List[Dict[str, Any]]) -> List[str]:
    """从数据中提取图片路径列表"""
    return [item['image'] for item in data]

def split_datasets_by_image_consistency(score_data: List[Dict[str, Any]], 
                                     deficiency_data: List[Dict[str, Any]], 
                                     train_ratio: float = 0.9, 
                                     random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    确保两个任务使用相同的图片分割
    
    Args:
        score_data: 评分数据
        deficiency_data: 缺陷数据
        train_ratio: 训练集比例
        random_seed: 随机种子
    
    Returns:
        (score_train, score_test, deficiency_train, deficiency_test)
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 获取所有唯一的图片路径
    score_images = set(get_image_paths(score_data))
    deficiency_images = set(get_image_paths(deficiency_data))
    
    # 检查两个数据集是否使用相同的图片
    if score_images != deficiency_images:
        print("警告: 两个数据集使用的图片不完全相同")
        print(f"Score数据集图片数量: {len(score_images)}")
        print(f"Deficiency数据集图片数量: {len(deficiency_images)}")
        print(f"共同图片数量: {len(score_images & deficiency_images)}")
    
    # 使用所有图片路径进行分割
    all_images = list(score_images | deficiency_images)
    random.shuffle(all_images)
    
    # 计算分割点
    total_size = len(all_images)
    train_size = int(total_size * train_ratio)
    
    # 分割图片路径
    train_images = set(all_images[:train_size])
    test_images = set(all_images[train_size:])
    
    # 根据图片路径分割数据
    score_train = [item for item in score_data if item['image'] in train_images]
    score_test = [item for item in score_data if item['image'] in test_images]
    deficiency_train = [item for item in deficiency_data if item['image'] in train_images]
    deficiency_test = [item for item in deficiency_data if item['image'] in test_images]
    
    return score_train, score_test, deficiency_train, deficiency_test

def main():
    """主函数"""
    print("开始数据集分割...")
    
    # 文件路径
    score_file = 'score_gpt.json'
    deficiency_file = 'deficiency_gpt.json'
    
    # 输出文件路径
    score_train_file = 'score_gpt_train.json'
    score_test_file = 'score_gpt_test.json'
    deficiency_train_file = 'deficiency_gpt_train.json'
    deficiency_test_file = 'deficiency_gpt_test.json'
    
    # 检查输入文件是否存在
    if not os.path.exists(score_file):
        print(f"错误: 文件 {score_file} 不存在")
        return
    
    if not os.path.exists(deficiency_file):
        print(f"错误: 文件 {deficiency_file} 不存在")
        return
    
    # 加载数据
    print("加载数据...")
    score_data = load_json_data(score_file)
    deficiency_data = load_json_data(deficiency_file)
    
    print(f"Score数据样本数: {len(score_data)}")
    print(f"Deficiency数据样本数: {len(deficiency_data)}")
    
    # 分割数据集，确保图片一致性
    print("分割数据集...")
    score_train, score_test, deficiency_train, deficiency_test = split_datasets_by_image_consistency(
        score_data, deficiency_data, train_ratio=0.9, random_seed=42
    )
    
    # 打印分割结果
    print(f"\n分割结果:")
    print(f"Score训练集: {len(score_train)} 样本")
    print(f"Score测试集: {len(score_test)} 样本")
    print(f"Deficiency训练集: {len(deficiency_train)} 样本")
    print(f"Deficiency测试集: {len(deficiency_test)} 样本")
    
    # 验证分割比例
    total_score = len(score_train) + len(score_test)
    total_deficiency = len(deficiency_train) + len(deficiency_test)
    score_train_ratio = len(score_train) / total_score if total_score > 0 else 0
    deficiency_train_ratio = len(deficiency_train) / total_deficiency if total_deficiency > 0 else 0
    
    print(f"\n实际分割比例:")
    print(f"Score训练集比例: {score_train_ratio:.3f}")
    print(f"Deficiency训练集比例: {deficiency_train_ratio:.3f}")
    
    # 保存分割后的数据
    print("\n保存分割后的数据...")
    save_json_data(score_train, score_train_file)
    save_json_data(score_test, score_test_file)
    save_json_data(deficiency_train, deficiency_train_file)
    save_json_data(deficiency_test, deficiency_test_file)
    
    print(f"已保存文件:")
    print(f"- {score_train_file}")
    print(f"- {score_test_file}")
    print(f"- {deficiency_train_file}")
    print(f"- {deficiency_test_file}")
    
    # 验证保存的文件
    print("\n验证保存的文件...")
    score_train_loaded = load_json_data(score_train_file)
    score_test_loaded = load_json_data(score_test_file)
    deficiency_train_loaded = load_json_data(deficiency_train_file)
    deficiency_test_loaded = load_json_data(deficiency_test_file)
    
    print(f"验证结果:")
    print(f"Score训练集加载: {len(score_train_loaded)} 样本")
    print(f"Score测试集加载: {len(score_test_loaded)} 样本")
    print(f"Deficiency训练集加载: {len(deficiency_train_loaded)} 样本")
    print(f"Deficiency测试集加载: {len(deficiency_test_loaded)} 样本")
    
    print("\n数据集分割完成！")

if __name__ == "__main__":
    main()
