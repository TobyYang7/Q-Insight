from huggingface_hub import snapshot_download

# 定义要下载的模型ID
# 模型ID格式通常是：作者/模型名称，例如 "bert-base-uncased"
# 或者 "meta-llama/Llama-2-7b"
model_id = "ByteDance/Q-Insight"

# 指定下载路径（可选，如果不指定则会下载到默认缓存目录）
local_dir = "./Q-Insight"

print(f"开始下载模型：{model_id}")

# 使用snapshot_download函数下载模型
# allow_patterns 参数可以指定只下载某些文件，比如只下载 PyTorch 权重文件
# ignore_patterns 参数可以指定不下载某些文件，比如不下载 TensorFlow 权重
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    # allow_patterns=["*.safetensors", "config.json"]  # 例如：只下载safetensors格式的权重和配置文件
    # ignore_patterns=["*.bin"] # 例如：不下载.bin文件
)

print(f"模型下载完成，保存在：{local_dir}")