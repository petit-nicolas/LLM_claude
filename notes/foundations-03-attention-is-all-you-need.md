# 基础知识 03：Attention Is All You Need 论文详解

> **论文**：Attention Is All You Need
> **作者**：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit 等 8 人（Google Brain / Google Research）
> **发表**：NeurIPS 2017 | arXiv: [1706.03762](https://arxiv.org/abs/1706.03762)
> **地位**：现代所有 LLM（GPT、BERT、LLaMA、DeepSeek…）的架构祖先
> **前置知识**：foundations-01-kv-cache.md（注意力公式部分）

---

## 一、论文要解决的问题

### 1.1 2017 年的 NLP 现状

论文发表时，序列到序列（seq2seq）任务（翻译、摘要等）的主流模型是：

```
RNN / LSTM 编码器-解码器 + Attention 机制
  ├── 编码器（Encoder）：逐个 Token 读入，压缩成隐状态向量
  └── 解码器（Decoder）：逐个 Token 生成，用 Attention 对齐编码器输出
```

### 1.2 RNN 的致命缺陷

**问题 1：无法并行**
```
RNN 的计算必须串行：
  h₁ = f(x₁)
  h₂ = f(x₂, h₁)   ← 必须等 h₁ 算完
  h₃ = f(x₃, h₂)   ← 必须等 h₂ 算完
  ...

GPU 有数千个核心，但 RNN 只能用一个 → 极大浪费
```

**问题 2：长距离依赖遗忘**
```
"The animal didn't cross the street because ____ was too tired."
  ↑                                              ↑
  想指代 animal，但信息要经过 N 个 RNN 步骤才能传递过来
  距离越远，梯度消失越严重，记忆越模糊
```

**问题 3：训练慢**
串行计算决定了序列越长，训练时间越长，无法有效扩展。

### 1.3 论文的核心主张

> **"Attention is all you need"**
> 完全抛弃 RNN 和 CNN，仅用注意力机制构建整个模型。

收益：
- 所有位置之间直接建立联系（路径长度 O(1)，而 RNN 是 O(N)）
- 高度并行，充分利用 GPU
- 实验结果超越所有前人工作

---

## 二、Transformer 整体架构

### 2.1 大局观

```
                    输入序列（英文）
                         │
                    ┌────▼────┐
                    │ Input   │  Token Embedding + Positional Encoding
                    │ Embed.  │
                    └────┬────┘
                         │
              ┌──────────▼──────────┐
              │    Encoder Stack    │  N=6 层，每层两个子层：
              │  ┌───────────────┐  │  1. Multi-Head Self-Attention
              │  │  Layer 1      │  │  2. Feed-Forward Network
              │  ├───────────────┤  │  + 残差连接 + LayerNorm
              │  │  Layer 2      │  │
              │  ├───────────────┤  │
              │  │     ...       │  │
              │  ├───────────────┤  │
              │  │  Layer 6      │  │
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │ encoder output（memory）
              ┌──────────▼──────────┐
              │    Decoder Stack    │  N=6 层，每层三个子层：
              │  ┌───────────────┐  │  1. Masked Multi-Head Self-Attention
              │  │  Layer 1      │  │  2. Cross-Attention（attend to encoder）
              │  ├───────────────┤  │  3. Feed-Forward Network
              │  │     ...       │  │  + 残差连接 + LayerNorm
              │  └───────────────┘  │
              └──────────┬──────────┘
                         │
                    ┌────▼────┐
                    │ Linear  │  投影到词表大小
                    │Softmax  │  输出概率分布
                    └────┬────┘
                         │
                    输出序列（中文）
```

### 2.2 模型超参数（Base 版本）

| 参数 | 值 |
|------|-----|
| 层数 N | 6 |
| 模型维度 d_model | 512 |
| FFN 中间维度 d_ff | 2048 |
| 注意力头数 h | 8 |
| 每头维度 d_k = d_v | 64（= 512/8） |
| Dropout | 0.1 |
| 参数量 | ~65M |

---

## 三、核心机制详解

### 3.1 缩放点积注意力（Scaled Dot-Product Attention）

这是整个论文最核心的操作：

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
```

**逐步分解：**

```
Step 1：计算 Query 和 Key 的相似度
  scores = Q × Kᵀ          shape: (seq_len, seq_len)
  scores[i][j] = "第 i 个 Token 对第 j 个 Token 的关注程度"

Step 2：缩放（防止梯度消失）
  scores = scores / √d_k
  ↑ 为什么要除以 √d_k？
    点积的方差随维度增大，d_k 大时 softmax 会饱和（梯度极小）
    除以 √d_k 使方差归一化

Step 3：Softmax 归一化
  weights = softmax(scores)   shape: (seq_len, seq_len)
  每行的权重之和 = 1

Step 4：加权求和
  output = weights × V        shape: (seq_len, d_v)
  每个位置的输出 = 所有 Value 的加权组合
```

**Mask 的作用：**
```
Decoder 中需要防止"看见未来"：
  在 softmax 之前，把未来位置的 score 设为 -∞
  softmax(-∞) = 0，等价于完全忽略未来 Token
```

### 3.2 多头注意力（Multi-Head Attention）

**为什么要多头？**
不同的注意力头可以关注不同的语义维度：
- 头 1：关注句法关系（主谓宾）
- 头 2：关注语义相似性
- 头 3：关注指代关系
- …

```python
# 多头注意力的计算过程（伪代码）
def multi_head_attention(Q, K, V, h=8, d_model=512):
    d_k = d_model // h  # = 64

    heads = []
    for i in range(h):
        # 每个头有独立的投影矩阵
        Qi = Q @ W_Q[i]   # (seq_len, d_k)
        Ki = K @ W_K[i]   # (seq_len, d_k)
        Vi = V @ W_V[i]   # (seq_len, d_v)

        head_i = attention(Qi, Ki, Vi)  # (seq_len, d_v)
        heads.append(head_i)

    # 拼接所有头的输出并投影回 d_model
    concat = torch.cat(heads, dim=-1)  # (seq_len, h*d_v = d_model)
    output = concat @ W_O              # (seq_len, d_model)
    return output
```

**数学表达：**
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W^O

其中 headᵢ = Attention(Q×Wᵢ^Q, K×Wᵢ^K, V×Wᵢ^V)
```

### 3.3 三种注意力的使用场景

```
┌───────────────────────────────────────────────────────────────┐
│ 1. Encoder Self-Attention（编码器自注意力）                     │
│    Q = K = V = 编码器当前层输出                                 │
│    作用：让每个输入 Token 看到所有其他输入 Token                 │
│    无 Mask：可以双向关注                                         │
└───────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────┐
│ 2. Decoder Masked Self-Attention（解码器掩码自注意力）           │
│    Q = K = V = 解码器当前层输出                                 │
│    作用：生成 Token 只能看到之前已生成的 Token                   │
│    有 Mask：单向（防止信息泄露）                                  │
└───────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────┐
│ 3. Encoder-Decoder Cross-Attention（交叉注意力）                │
│    Q = 解码器当前层输出                                          │
│    K = V = 编码器最终输出（memory）                              │
│    作用：解码时对齐、查询源语言信息                               │
│    无 Mask：解码器可以看到完整的编码器输出                        │
└───────────────────────────────────────────────────────────────┘
```

### 3.4 Position-wise Feed-Forward Network（FFN）

每个注意力层后面都接一个 FFN：

```
FFN(x) = max(0, x × W₁ + b₁) × W₂ + b₂
         └───────────────────┘
               ReLU 激活

维度变化：d_model(512) → d_ff(2048) → d_model(512)

作用：
  - 注意力机制负责"聚合信息"（跨 Token 交互）
  - FFN 负责"处理信息"（逐位置的非线性变换）
  - 两者分工合作
```

**现代 LLM 的改进**：用 SwiGLU 替代 ReLU：
```
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
↑ LLaMA、DeepSeek 等都用这个，效果更好
```

### 3.5 残差连接（Residual Connection）与 Layer Normalization

每个子层都包裹了：
```
output = LayerNorm(x + Sublayer(x))

残差连接的作用：
  1. 解决深层网络梯度消失问题
  2. 允许信息直接"跳过"某些层
  3. 类似高速公路（Highway Networks）

LayerNorm 的作用：
  1. 稳定训练过程
  2. 在特征维度上归一化（不同于 BatchNorm 在批次维度）
```

**注意**：原论文是 Post-LN（归一化在残差之后），现代 LLM 普遍改用 **Pre-LN**（归一化在子层之前），训练更稳定。

### 3.6 位置编码（Positional Encoding）

注意力机制本身没有位置信息（词序无关），需要额外注入位置信息：

**原论文使用正弦/余弦函数：**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
  pos = Token 在序列中的位置（0, 1, 2, ...）
  i   = 维度索引（0, 1, ..., d_model/2 - 1）
```

**直觉解释：**
```
不同维度的频率不同：
  低维（i≈0）：频率高，变化快 → 区分相邻位置
  高维（i≈d_model/2）：频率低，变化慢 → 区分远距离位置

类比：
  二进制计数 001, 010, 011, 100...
  低位快速变化，高位缓慢变化
  每个位置有唯一的"二进制编码"
```

**现代改进**：
- **RoPE（旋转位置编码）**：LLaMA、DeepSeek 等用这个，支持外推到更长上下文
- **ALiBi**：通过注意力 bias 编码位置，不改变向量本身

---

## 四、训练细节

### 4.1 优化器

使用 Adam 优化器，但学习率使用特殊调度：

```
lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))

含义：
  前 warmup_steps 步：线性增大（warmup 阶段）
  之后：按 step^(-0.5) 递减

原论文：warmup_steps = 4000
```

**为什么要 Warmup？**
- 训练初期参数随机，梯度不稳定
- 从小学习率开始，逐渐增大，避免早期训练崩溃

### 4.2 正则化

| 方法 | 说明 |
|------|------|
| Dropout | 每个子层的输出、词嵌入、位置编码后均施加 |
| Label Smoothing | 目标分布不是 one-hot，而是 0.1 的 smoothing |

**Label Smoothing 的直觉：**
```
不平滑：正确标签概率 = 1.0，其他 = 0
平滑后：正确标签概率 = 0.9，其他 = 0.1/|V| 均匀分布
作用：防止模型过于自信，提升 BLEU 分数
```

### 4.3 训练数据与硬件

| 任务 | 数据集 | 数据量 | 硬件 | 训练时间 |
|------|-------|-------|------|---------|
| EN→DE 翻译 | WMT 2014 | 4.5M 句对 | 8 × P100 GPU | 12 小时（base）/ 3.5 天（big） |
| EN→FR 翻译 | WMT 2014 | 36M 句对 | 8 × P100 GPU | 3.5 天（big） |

---

## 五、实验结果

### 5.1 机器翻译（BLEU 分数）

| 模型 | EN→DE | EN→FR | 参数量 | 训练成本（FLOPs） |
|------|-------|-------|-------|-----------------|
| ByteNet | 23.75 | — | — | — |
| Deep-Att + PosUnk | — | 39.2 | — | — |
| GNMT + RL | 24.6 | 41.16 | — | 大 |
| ConvS2S | 25.16 | 40.46 | — | 大 |
| **Transformer (Big)** | **28.4** | **41.8** | 213M | 小得多 |
| **Transformer (Base)** | **27.3** | **38.1** | 65M | 更小 |

**结论**：Transformer 在 BLEU 上比所有前人工作高出 2 BLEU 以上，且训练时间大幅缩短。

### 5.2 消融实验（Ablation Study）

论文通过修改单个超参数，验证各组件的贡献：

| 变体 | EN→DE BLEU | 备注 |
|------|-----------|------|
| (A) h=8 heads（原始） | 25.8 | 基准 |
| (A) h=1（单头） | 23.3 | 多头的必要性 |
| (A) h=32，d_k=16 | 25.5 | 头太多反而略差 |
| (B) d_k=16（太小） | 25.5 | 降低质量 |
| (C) 更大模型 | 26.4 | 规模扩展有效 |
| (D) Dropout=0 | 25.3 | Dropout 有帮助 |
| (E) 位置编码改学习 | 25.8 | 与正弦编码相近 |

### 5.3 英语成分句法分析

为验证泛化能力，论文把 Transformer 用于句法分析任务（与翻译完全不同），结果超越了专门为此设计的模型，证明架构的通用性。

---

## 六、为什么这篇论文如此重要？

### 6.1 直接影响

```
Attention Is All You Need (2017)
    │
    ├── BERT (2018)：双向 Encoder，语言理解
    │       └── RoBERTa, ALBERT, DeBERTa...
    │
    ├── GPT (2018)：单向 Decoder，语言生成
    │       └── GPT-2, GPT-3, GPT-4, Claude, LLaMA...
    │                                    └── DeepSeek, Qwen, Mistral...
    │
    └── T5 (2019)：编码器-解码器，统一框架
            └── mT5, Flan-T5, BART...
```

### 6.2 三个关键创新的深远影响

**1. 纯注意力 → 高度并行化**
- 使大规模训练成为可能
- 奠定了 Scaling Law 的基础

**2. 全局感受野**
- 任意两个 Token 之间 O(1) 路径长度
- LLM 能处理复杂的长距离依赖

**3. 通用架构**
- 同一套架构应用于 NLP、CV（ViT）、语音（Whisper）、科学计算（AlphaFold）…

### 6.3 与 KV-Cache 的连接

原论文没有 KV-Cache 的概念（当时是翻译任务，不是自回归生成）。但 GPT 系列使用 Decoder-only 架构做自回归生成时：
- 每步生成只产出一个 Token
- 历史 Token 的 K、V 不需要重新计算
- **KV-Cache 自然地从这个架构特性中产生**

---

## 七、从这篇论文到 DeepSeek / DualPath

```
Attention Is All You Need（2017）
  ↓ 奠定 Transformer 架构
GPT 系列（2018-2023）
  ↓ Decoder-only，自回归生成，催生 KV-Cache 需求
LLaMA / DeepSeek 等（2023-2026）
  ↓ 超大模型，超长上下文，KV-Cache 体积爆炸
  ↓ KV-Cache 需要存到 SSD，加载成为瓶颈
DualPath（2026）
  ↓ 通过双路加载突破存储带宽瓶颈
```

理解了 Transformer 的注意力机制，就理解了：
- 为什么会有 K 和 V（注意力的两个要素）
- 为什么 KV-Cache 和层数、头数成正比（每层独立的 K/V）
- 为什么 GQA 能减小 KV-Cache（减少 K/V 头数）

---

## 八、关键概念速查表

| 概念 | 公式/含义 |
|------|---------|
| Scaled Dot-Product Attention | `softmax(QKᵀ/√d_k) × V` |
| Multi-Head Attention | 多个独立注意力头并行，结果拼接再投影 |
| d_model | 模型的统一隐藏维度（Base: 512, Big: 1024） |
| d_k / d_v | 每头的 Key/Value 维度 = d_model / h |
| d_ff | FFN 的中间维度（= 4 × d_model） |
| h | 注意力头数（Base: 8, Big: 16） |
| N | Encoder/Decoder 的层数（= 6） |
| Warmup | 学习率从小逐渐增大，再按 step^(-0.5) 衰减 |
| Label Smoothing | 软化目标概率分布，防止过拟合 |
| Pre-LN vs Post-LN | 现代 LLM 用 Pre-LN，训练更稳定 |
| RoPE | 旋转位置编码，现代 LLM 标配 |

---

## 九、自测问题

1. 为什么注意力分数要除以 √d_k？不除会怎样？
2. Encoder 和 Decoder 的 Self-Attention 有什么区别？为什么 Decoder 需要 Mask？
3. FFN 的输入和输出维度都是 d_model，中间是 d_ff（4×d_model），这种设计有什么好处？
4. 原论文用正弦位置编码，现代 LLM 改用 RoPE，改动的主要动机是什么？
5. 消融实验说明多头数量越多越好吗？为什么？

> 答案提示：
> 1. d_k 大时点积方差大，softmax 饱和，梯度接近 0，训练困难
> 2. Encoder 是双向（可看全局），Decoder 是单向（生成时不能看未来），用 Mask 遮盖
> 3. 先升维（非线性表达能力）再降维（信息压缩），类似 bottleneck 结构
> 4. 正弦编码外推能力差（超过训练长度效果急剧下降），RoPE 外推性更好
> 5. 不是，h=32 时 d_k=16 太小，每头表达能力受限，反而略差于 h=8

---

## 参考链接

- [原始论文（arXiv）](https://arxiv.org/abs/1706.03762)
- [论文 PDF 直链](https://arxiv.org/pdf/1706.03762)
- [The Illustrated Transformer（最佳图解）](http://jalammar.github.io/illustrated-transformer/)
- [Harvard NLP 注释实现](https://nlp.seas.harvard.edu/annotated-transformer/)
