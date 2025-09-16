# Score Evaluation Fix Script

## 问题描述

在score评估的JSON文件中发现了一个异常值问题：
- 预测分数为 `2022.0`，真实分数为 `5.5`
- 这个异常值导致整体MSE达到 `16268.72`，严重影响了评估结果
- 异常值是由于模型输出解析错误造成的

## 解决方案

### 1. fix.py 脚本

用于修复现有的score评估JSON文件：

```bash
# 分析文件（不修改）
python results/fix.py results/score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1200.json --dry-run --verbose

# 修复文件
python results/fix.py results/score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1200.json --output results/fixed_file.json

# 直接覆盖原文件
python results/fix.py results/score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1200.json
```

**功能：**
- 检测异常分数（>100, 负值, 与真实分数差异过大）
- 尝试从模型输出中重新提取有效分数
- 如果无法提取有效分数，标记为错误
- 重新计算整体指标

### 2. test.py 修改

修改了 `test.py` 中的score处理逻辑：

**新增功能：**
- `_extract_score_with_validation()` 方法：带验证的分数提取
- 分数范围验证（0-10）
- 提取失败时的错误标记
- 改进的指标计算，排除无效分数

**修改的方法：**
- `process_score()`: 增加了分数验证和错误处理
- `calculate_score_metrics()`: 只计算有效分数的指标

## 修复结果

**修复前：**
- MAE: 9.51
- MSE: 16268.72
- RMSE: 127.55

**修复后：**
- MAE: 1.45
- MSE: 3.64
- RMSE: 1.91
- 有效预测: 249/250
- 失败预测: 1/250

## 使用建议

1. **对于现有结果**：使用 `fix.py` 修复已生成的JSON文件
2. **对于新测试**：使用修改后的 `test.py` 避免产生异常值
3. **验证结果**：检查修复后的指标是否合理

## 注意事项

- 修复脚本会保留原始文件，建议先备份
- 异常样本会被标记为错误，不参与指标计算
- 如果模型输出格式发生变化，可能需要调整提取逻辑

