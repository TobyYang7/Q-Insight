<div align="center">
<h3>

Q-Insight: Understanding Image Quality via Visual Reinforcement Learning
</h3>

  <a href="https://arxiv.org/abs/2503.22679">
    <img
      src="https://img.shields.io/badge/QInsight-Paper-red?logo=arxiv&logoColor=red"
      alt="Q-Insight Paper on arXiv"
    />
  </a>
<a href="https://huggingface.co/ByteDance/Q-Insight">
    <img 
        src="https://img.shields.io/badge/QInsight-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="Q-Insight Model"
    />
</a>


[Weiqi Li](https://scholar.google.com/citations?user=SIkQdEsAAAAJ), Xuanyu Zhang, Shijie Zhao, Yabin Zhang, Junlin Li, Li Zhang and [Jian Zhang](https://jianzhang.tech/)
</div>

## 🔥 Introduction
PLCC comparisons between our proposed Q-Insight and existing IQA metrics (left) and three example applications of our Q-Insight (right) are presented. Q-Insight demonstrates significantly improved performance compared to existing methods, especially on out-of-domain datasets. Additionally, Q-Insight effectively supports quality score regression, image degradation perception, and zero-shot image comparison reasoning tasks.
<p align="center">
  <img src="assets/teaser.png">
</p>

## 🚩 Updates
- 05.30 Released training and testing code, along with the pretrained model.
- 05.26 Released our v2 paper.
- 03.28 Released the Q-Insight technical report.

## 🔧 Dependencies and Installation
```bash
git clone https://github.com/bytedance/Q-Insight.git
bash setup.sh
```

## ⚡ Quick Inference
### Supported Tasks
#### Score Regression
```
cd src/eval/
python demo_score.py
```
#### Degradation Perception
```
cd src/eval/
python demo_dist.py
```
#### Image Comparison Reasoning
```
cd src/eval/
python demo_comparison.py
```

## 📖 Dataset Preparation for Training
#### Score Regression
Download meta files from [Data-DeQA-Score](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main) and the source images from the [KONIQ](https://database.mmsp-kn.de/koniq-10k-database.html) dataset.
Arrange the folders in `./src/open-r1-multimodal/data`as follows:
```
|-- Data-DeQA-Score
  |-- KONIQ
    |-- images/*.jpg
    |-- metas
```
#### Degradation Perception
Download the `refA_sd_brief` subset from [KADIS-700K](https://modelscope.cn/datasets/zhiyuanyou/DataDepictQA/files).
Arrange the folders in `./src/open-r1-multimodal/data` as follows:
```
|-- KADIS-700K
  |-- refA_sd_brief
    |-- dist_imgs/*.jpg
    |-- metas/train_dist.json
```

#### Image Comparison Reasoning
Download the validation dataset of [DiffIQA](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht).
Arrange the folders in `./src/open-r1-multimodal/data` as follows:
```
|-- DiffIQA
  |-- ValidationImage
    |-- images
    |-- train_comparison.json
```

## Training
#### Score Regression and Degradation Perception
```
cd src/open-r1-multimodal/
bash run_qinsight_score_and_dist.sh
```
#### Image Comparison Reasoning
```
cd src/open-r1-multimodal/
bash run_qinsight_comparison.sh
```


## ✏️ To Do List
- [ ] Add support for LoRA fine-tuning
- [ ] Provide a Gradio demo
- [x] Release inference code and weights
- [x] Release training code
- [x] Release the paper

## Acknowledgement
We appreciate the releasing codes and data of [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [DepictQA](https://github.com/XPixelGroup/DepictQA) and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).



## Citation
If Q-Insight is helpful, please help to ⭐ the repo.

If you find the code helpful in your research or work, please cite the following paper:
```
@article{li2025qinsight,
  title={Q-Insight: Understanding Image Quality via Visual Reinforcement Learning},
  author={Li, Weiqi and Zhang, Xuanyu and Zhao, Shijie and Zhang, Yabin and Li, Junlin and Zhang, Li and Zhang, Jian},
  journal={arXiv preprint arXiv:2503.22679},
  year={2025}
}
```
