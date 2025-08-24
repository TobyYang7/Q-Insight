git clone https://github.com/TobyYang7/paper-slide-crawler
git clone https://github.com/zhuohaouw/SlideAudit
git config --global user.email "tobyyang7@outlool.com"
git config --global user.name "TobyYang7"
pip install nvitop
pip install uv
uv venv

cd src/open-r1-multimodal 
uv pip install -e ".[dev]"

# Addtional modules
uv pip install wandb==0.18.3
uv pip install tensorboardx
uv pip install qwen_vl_utils torchvision
uv pip install flash-attn --no-build-isolation
uv pip install transformers==4.51.3
