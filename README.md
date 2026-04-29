# CLIP Prompt Tuning for Remote Sensing Image-Text Retrieval

> 用 4,096 个可训练参数（CLIP 主干的 0.003%）把 RSITMD 遥感图文检索 R@1 相对提升 74%

## 项目背景

OpenAI CLIP 在自然图像上零样本表现强，但训练数据以日常照片为主，
对**俯视角遥感影像**理解能力弱。本项目验证：能否用极少量参数（PEFT）
把 CLIP 迁移到遥感场景，避免全参微调的算力与过拟合代价。

## 方法

- **Backbone**：OpenAI CLIP ViT-B/32，全部 ≈ 151M 参数冻结
- **Prompt Tuning**：CoOp 风格，文本端 SOS 之后插入 8 个可学习软 prompt token
- **可训练参数**：4,096（仅占主干 0.003%）
- **损失**：双向 InfoNCE
- **优化**：AdamW（lr 5e-4）+ Cosine 退火，20 epoch

## 实验结果（RSITMD test）

| 方向 | 方法 | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| 图 → 文 | Zero-shot CLIP    | 8.628318458795547 | 22.123894095420837 | 32.0796459913253 |
| 图 → 文 | + Prompt Tuning   | **14.823009073734283** | 34.292036294937134 | 47.56637215614319 |
| 文 → 图 | Zero-shot CLIP    | 7.92035385966301 | 25.79646110534668 |  41.7256623506546 |
| 文 → 图 | + Prompt Tuning   | **11.017698794603348** | 38.84955644607544 | 58.36282968521118 |
主要结论：仅用 4,096 个参数，**图→文 R@1 相对提升 74%**，文→图 R@1 相对提升 39%。

![Training curve](assets/training_curve.png)
![GradCAM comparison](assets/gradcam_comparison.png)

## 工程踩坑（值得记录的两个真实 bug）

### 1. 新版 open_clip Transformer 默认 `batch_first=True`
原始 CoOp 代码在文本编码时显式做 `permute(1,0,2)` 把张量转成 `[L,B,D]`。
新版 open_clip 的 Transformer 默认 `batch_first=True`（即 `[B,L,D]`），
强行 permute 会导致 attn_mask 形状错位（`[77,77]` vs 期望 `[B,B]`）。

**修复**：通过 `getattr(transformer, 'batch_first', True)` 自适应判断。

### 2. CLIP 全参冻结时 GradCAM 反向传播链断裂
`Prompt Tuning` 把 CLIP 全部参数 `requires_grad=False`。GradCAM 默认
不会自动给输入张量打 `requires_grad`，导致 autograd 链整条断掉，
`loss.backward()` 报 `does not have a grad_fn`。

**修复**：在送入 CAM 前显式 `inp.requires_grad_(True)`。

## Quick Start

\`\`\`bash
pip install open_clip_torch torch torchvision tifffile grad-cam
\`\`\`

\`\`\`python
import torch, open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
ckpt = torch.load('models/best_prompt.pt')
# 将 ckpt['ctx'] 加载到自己的 PromptCLIP 实例中即可推理
\`\`\`

完整训练与评估流程见 `notebooks/CLIP_PromptTuning.ipynb`。

## 数据

- [RSITMD](https://github.com/xiaoyuan1996/AMFMN)：4,743 张遥感图像，每张 5 条人工标注描述

## 参考

- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021
- Zhou et al. *Learning to Prompt for Vision-Language Models* (CoOp), IJCV 2022
