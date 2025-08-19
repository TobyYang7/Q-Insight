cd src/open-r1-multimodal 
uv pip install -e ".[dev]"

# Addtional modules
uv pip install wandb==0.18.3
uv pip install tensorboardx
uv pip install qwen_vl_utils torchvision
uv pip install flash-attn --no-build-isolation
uv pip install transformers==4.51.3
