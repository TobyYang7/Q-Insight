# Score JSON 文件修复报告

## 修复概览

成功修复了所有以 `score` 开头的JSON文件，共处理了 **6个文件**。

## 修复结果详情

### 1. score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1000.json
- **状态**: ✅ 无异常值
- **修复前**: MAE: 1.5236, MSE: 3.9619, RMSE: 1.9905
- **修复后**: MAE: 1.5236, MSE: 3.9619, RMSE: 1.9905
- **变化**: 无变化

### 2. score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1200.json
- **状态**: ✅ 发现并修复1个异常值
- **异常值**: 2022.0 (GT: 5.5) → 标记为错误
- **修复前**: MAE: 9.5088, MSE: 16268.72, RMSE: 127.55
- **修复后**: MAE: 1.4486, MSE: 3.6439, RMSE: 1.9089
- **改进**: MAE降低85%, MSE降低99.98%

### 3. score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt200.json
- **状态**: ✅ 无异常值
- **修复前**: MAE: 1.8806, MSE: 5.8066, RMSE: 2.4097
- **修复后**: MAE: 1.8806, MSE: 5.8066, RMSE: 2.4097
- **变化**: 无变化

### 4. score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt600.json
- **状态**: ✅ 发现并修复2个异常值
- **异常值**: 
  - 2023.0 (GT: 7.5) → 修复为 9.5
  - 5879.0 (GT: 7.5) → 修复为 3.0
- **修复前**: MAE: 33.37, MSE: 154771.71, RMSE: 393.41
- **修复后**: MAE: 1.7194, MSE: 5.7289, RMSE: 2.3935
- **改进**: MAE降低94.8%, MSE降低99.6%

### 5. score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt800.json
- **状态**: ✅ 无异常值
- **修复前**: MAE: 1.6390, MSE: 4.3789, RMSE: 2.0926
- **修复后**: MAE: 1.6390, MSE: 4.3789, RMSE: 2.0926
- **变化**: 无变化

### 6. score_Qwen2.5-VL-7B-Instruct.json
- **状态**: ✅ 无异常值
- **修复前**: MAE: 1.3747, MSE: 3.2854, RMSE: 1.8126
- **修复后**: MAE: 1.3747, MSE: 3.2854, RMSE: 1.8126
- **变化**: 无变化

## 修复统计

- **总文件数**: 6
- **发现异常值的文件**: 2
- **总异常值数量**: 3
- **成功修复**: 2
- **标记为错误**: 1

## 关键发现

1. **ckpt1200** 和 **ckpt600** 存在严重的异常值问题
2. 异常值主要是由于模型输出解析错误造成的
3. 修复后所有模型的指标都回到了合理范围
4. 大部分文件（4/6）本身没有异常值问题

## 建议

1. 使用修复后的文件进行后续分析
2. 在模型训练过程中增加输出格式验证
3. 定期检查评估结果中的异常值
4. 考虑在测试流程中集成异常值检测

## 文件列表

修复后的文件都添加了 `_fixed` 后缀：
- `score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1000_fixed.json`
- `score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt1200_fixed.json`
- `score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt200_fixed.json`
- `score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt600_fixed.json`
- `score_eval_multi_ep3_f1_0.6_s_0.3_all-ckpt800_fixed.json`
- `score_Qwen2.5-VL-7B-Instruct_fixed.json`

