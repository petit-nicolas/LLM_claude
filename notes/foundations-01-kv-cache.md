# 基础知识 01：KV-Cache 详解

> 关联论文：DualPath（arXiv:2602.21548）
> 学习目标：理解 KV-Cache 的来源、作用、结构与代价

---

## 一、从 Transformer 注意力机制说起

### 1.1 Attention 的基本公式

Transformer 的核心是 **自注意力机制（Self-Attention）**：

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d) × V
```

三个矩阵的含义：

| 矩阵 | 全称 | 含义 |
|------|------|------|
| **Q** | Query（查询） | "我想找什么信息？" |
| **K** | Key（键） | "每个位置能提供什么信息？" |
| **V** | Value（值） | "实际的信息内容是什么？" |

**直觉类比**：就像在图书馆检索书目
- Q：你的检索关键词
- K：每本书的目录标签
- V：书的实际内容
- Attention Score：你的关键词与每本书标签的匹配度

### 1.2 多头注意力（Multi-Head Attention）

实际中会并行运行 H 个注意力头，每个头学习不同的关注维度：

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₙ) × Wᵒ
其中 headᵢ = Attention(Q×Wᵢᵠ, K×Wᵢᴷ, V×Wᵢᵛ)
```

---

## 二、为什么需要 KV-Cache？

### 2.1 自回归生成的本质

LLM 生成文本是**自回归（Autoregressive）**的：每次只生成一个新 Token，然后把它加入输入，再生成下一个。

```
输入：     [The][cat][sat]
生成第1步：[The][cat][sat] → "on"
生成第2步：[The][cat][sat][on] → "the"
生成第3步：[The][cat][sat][on][the] → "mat"
```

### 2.2 没有 KV-Cache 时的重复计算

每次生成新 Token，注意力机制需要对**所有历史 Token** 重新计算 K 和 V：

```
第1步（生成 "on"）：
  计算 K,V 用于：[The, cat, sat]                  ← 3次计算

第2步（生成 "the"）：
  计算 K,V 用于：[The, cat, sat, on]              ← 4次计算（前3次重复！）

第3步（生成 "mat"）：
  计算 K,V 用于：[The, cat, sat, on, the]         ← 5次计算（前4次重复！！）
```

对于长度 N 的序列，总计算量是 O(N²)，大量计算被重复执行。

### 2.3 KV-Cache 的核心思想

**把已经计算过的 K 和 V 缓存起来，下次直接复用。**

```
第1步：计算并缓存 K,V for [The, cat, sat]
        cache = {The:K₁V₁, cat:K₂V₂, sat:K₃V₃}

第2步：只计算新 Token "on" 的 K,V
        新建：on:K₄V₄
        拼接：cache = {The,cat,sat,on 的 K,V}

第3步：只计算新 Token "the" 的 K,V
        ...以此类推
```

**效果**：每步生成只需 O(1) 新计算，历史 Token 的 K,V 全部复用，节省了大量算力。

---

## 三、KV-Cache 的物理结构

### 3.1 每层都有独立的 KV-Cache

现代 LLM 是多层 Transformer 叠加的。以 32 层模型为例：

```
Layer 1:  KV-Cache for all tokens
Layer 2:  KV-Cache for all tokens
...
Layer 32: KV-Cache for all tokens
```

每一层的注意力计算都需要访问该层的历史 KV，所以每层独立缓存。

### 3.2 KV-Cache 的大小计算

```
KV-Cache 大小 = 2 × 层数 × 注意力头数 × 头维度 × 序列长度 × 数据类型字节数

以 LLaMA-2 70B 为例（BF16精度）：
  - 层数：80 层
  - KV 头数：8（GQA）
  - 头维度：128
  - 一个 Token 的 KV-Cache：2 × 80 × 8 × 128 × 2 字节 = 327,680 字节 ≈ 320 KB
  - 1万 Token 的上下文：约 3.2 GB
  - 10万 Token（Agent 长任务）：约 32 GB！！
```

这说明为什么长上下文 Agent 推理中 KV-Cache 会变得巨大。

### 3.3 GQA：节省 KV-Cache 的设计

标准 Multi-Head Attention (MHA) 每个 Q head 都有对应的 K/V head，开销最大。

**Group Query Attention (GQA)**：多个 Q head 共享同一组 K/V，大幅压缩 KV-Cache 体积：

```
MHA：  Q heads = K heads = V heads = 32  （1:1:1）
GQA：  Q heads = 32, K heads = V heads = 8 （4:1:1，缩小4倍）
MQA：  Q heads = 32, K heads = V heads = 1 （极端压缩）
```

DeepSeek V3 还用了 **MLA（Multi-head Latent Attention）**，通过低秩压缩进一步减小 KV-Cache 体积。

---

## 四、KV-Cache 的存储层级

### 4.1 三级存储金字塔

```
                    ┌───────────┐
                    │    HBM    │  GPU 显存
                    │  快但小   │  容量：几十到百GB
                    │  ~3TB/s   │  当前请求的 KV-Cache
                    └─────┬─────┘
                          │  溢出时卸载
                    ┌─────▼─────┐
                    │   DRAM    │  CPU 内存
                    │  中等速度  │  容量：几百GB至TB级
                    │  ~100GB/s │  近期请求的 KV-Cache
                    └─────┬─────┘
                          │  溢出时卸载
                    ┌─────▼─────┐
                    │  SSD/NVMe │  外部存储
                    │  慢但大   │  容量：几十TB
                    │  ~10GB/s  │  历史 KV-Cache（跨轮复用）
                    └───────────┘
```

### 4.2 为什么 Agent 场景需要 SSD 级存储

在 Agent 多轮对话中：
- 每轮对话完成后，KV-Cache 存入 SSD
- 下一轮开始时，从 SSD 读取历史 KV-Cache（Cache Hit）
- 命中率极高（生产环境 >95%），大部分 Token 无需重新 Prefill

但 SSD 的带宽远低于 HBM，加载大量 KV-Cache 就成为系统瓶颈——这正是 DualPath 论文要解决的核心问题。

---

## 五、KV-Cache 命中（Prefix Caching）

### 5.1 Prefix Cache 的工作原理

如果两个请求有相同的前缀（System Prompt、历史对话等），它们的 KV-Cache 可以共享：

```
请求 A：[系统提示][用户问题A][模型回答A][用户问题B...]
请求 B：[系统提示][用户问题A][模型回答A][用户问题C...]
                 └────────────────────┘
                      相同部分的 KV-Cache 可以复用
```

### 5.2 为什么 Agent 场景命中率极高

Agent 轨迹的典型结构：

```
轮次1：System Prompt + 用户指令
轮次2：System Prompt + 用户指令 + [工具调用结果1]
轮次3：System Prompt + 用户指令 + [工具调用结果1] + [工具调用结果2]
...
轮次N：System Prompt + 用户指令 + [N-1条历史记录]
```

每一轮的新增内容相对于总上下文非常小，因此历史 KV-Cache 的复用率接近 100%。

---

## 六、KV-Cache 的代价与挑战

### 6.1 显存占用（内存墙）

| 问题 | 说明 |
|------|------|
| 显存被 KV-Cache 占满 | 大 batch 或长上下文场景下，KV-Cache 可能比模型权重还大 |
| 显存碎片化 | 不同请求长度不一，导致显存分配碎片 |
| vLLM 的解法 | PagedAttention：把 KV-Cache 分成固定大小的"页"，类似操作系统内存管理 |

### 6.2 IO 带宽（IO 墙）

在 SSD 级 KV-Cache 场景：

```
问题链：
  Agent 长上下文 → KV-Cache 巨大（几十GB）
  → 每轮开始需要从 SSD 加载大量数据
  → SSD 带宽（~10GB/s）远低于 GPU 计算速度
  → GPU 空等 IO → 吞吐下降

DualPath 的切入点：如何更快地把 KV-Cache 从 SSD 搬到 GPU
```

### 6.3 带宽趋势（DualPath 论文数据）

从 Ampere 到 Blackwell，GPU 算力增长远超 IO 带宽增长，IO/Compute 比例下降 14.4×，IO 墙问题将持续恶化。

---

## 七、本节小结与与 DualPath 的连接

```
Transformer 注意力
    ↓
自回归生成 → 历史 K/V 重复计算 → 引入 KV-Cache
    ↓
KV-Cache 体积随上下文增长 → 需要多级存储
    ↓
Agent 场景：上下文极长 + 跨轮复用 → KV-Cache 主要在 SSD
    ↓
SSD → GPU 的 IO 带宽成为推理瓶颈
    ↓
DualPath：通过双路加载 + 全局调度突破这个带宽瓶颈
```

---

## 八、自测问题

1. 为什么 KV-Cache 只缓存 K 和 V，而不缓存 Q？
2. 如果一个模型有 96 层、GQA 8 头、头维度 128，用 FP16，10k Token 的 KV-Cache 有多大？
3. 为什么 Agent 场景的 KV-Cache 命中率特别高？
4. MHA 和 GQA 在 KV-Cache 大小上有什么区别？

> 答案提示：
> 1. Q 是当前生成 Token 的查询向量，每次都是新的，无法复用
> 2. 2 × 96 × 8 × 128 × 2B × 10000 = 约 3.93 GB
> 3. 多轮对话中历史内容占比极高，新增内容只占一小部分
> 4. GQA 用更少的 K/V 头，KV-Cache 是 MHA 的 (kv_heads/q_heads) 倍大小
