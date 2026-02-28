# 论文资源索引

记录本项目涉及的所有重要论文，包含下载链接和本地存储路径。
使用 `download_papers.sh` 脚本一次性下载所有论文 PDF。

---

## 核心论文

### Transformer 基础

| 论文 | 作者 | 年份 | arXiv | 本地文件 | 状态 |
|------|------|------|-------|---------|------|
| Attention Is All You Need | Vaswani et al. | 2017 | [1706.03762](https://arxiv.org/abs/1706.03762) | `attention-is-all-you-need.pdf` | 待下载 |
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | 2018 | [1810.04805](https://arxiv.org/abs/1810.04805) | `bert-1810.04805.pdf` | 待下载 |
| Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 2020 | [2005.14165](https://arxiv.org/abs/2005.14165) | `gpt3-2005.14165.pdf` | 待下载 |

### LLM 推理优化

| 论文 | 作者 | 年份 | arXiv | 本地文件 | 状态 |
|------|------|------|-------|---------|------|
| FlashAttention | Dao et al. | 2022 | [2205.14135](https://arxiv.org/abs/2205.14135) | `flashattention-2205.14135.pdf` | 待下载 |
| FlashAttention-2 | Dao | 2023 | [2307.08691](https://arxiv.org/abs/2307.08691) | `flashattention2-2307.08691.pdf` | 待下载 |
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al. | 2023 | [2309.06180](https://arxiv.org/abs/2309.06180) | `pagedattention-2309.06180.pdf` | 待下载 |
| GQA: Training Generalized Multi-Query Transformer | Ainslie et al. | 2023 | [2305.13245](https://arxiv.org/abs/2305.13245) | `gqa-2305.13245.pdf` | 待下载 |

### DeepSeek 系列

| 论文 | 作者 | 年份 | arXiv | 本地文件 | 状态 |
|------|------|------|-------|---------|------|
| DeepSeek-V3 Technical Report | DeepSeek | 2024 | [2412.19437](https://arxiv.org/abs/2412.19437) | `deepseek-v3-2412.19437.pdf` | 待下载 |
| DualPath: Breaking the Storage Bandwidth Bottleneck | DeepSeek et al. | 2026 | [2602.21548](https://arxiv.org/abs/2602.21548) | `dualpath-2602.21548.pdf` | 待下载 |

### PD 分离 & Agentic 推理

| 论文 | 作者 | 年份 | arXiv | 本地文件 | 状态 |
|------|------|------|-------|---------|------|
| Sarathi-Serve: Chunked Prefills for LLM Inference | Agrawal et al. | 2024 | [2403.02310](https://arxiv.org/abs/2403.02310) | `sarathi-serve-2403.02310.pdf` | 待下载 |
| Mooncake: A KVCache-centric Disaggregated Architecture | Qin et al. | 2024 | [2407.00079](https://arxiv.org/abs/2407.00079) | `mooncake-2407.00079.pdf` | 待下载 |

---

## 下载说明

在有外部网络访问权限的环境中运行：

```bash
cd /home/user/LLM_claude/resources
bash download_papers.sh
```
