# 基础知识 02：Prefill 与 Decode 两阶段推理

> 关联论文：DualPath（arXiv:2602.21548）
> 前置知识：foundations-01-kv-cache.md
> 学习目标：理解 LLM 推理的两个阶段，以及 PD 分离架构的来源与意义

---

## 一、LLM 推理的两个阶段

### 1.1 直觉理解

想象你在回答一道问答题：
- **读题阶段**：一次性读完整个题目，理解所有信息
- **答题阶段**：一个字一个字地写出答案，每写一个字都要考虑前面写的内容

LLM 推理完全一样：

| 阶段 | 英文 | 类比 | 特点 |
|------|------|------|------|
| **Prefill** | 预填充 | 读题 | 一次处理所有输入 Token |
| **Decode** | 解码 | 答题 | 每次只生成一个 Token |

### 1.2 一个完整的推理过程

```
用户输入：「请用一句话介绍中国的首都」（12个Token）

━━━━━━━━━━━━━━ Prefill 阶段 ━━━━━━━━━━━━━━
输入：[请][用][一][句][话][介][绍][中][国][的][首][都]
      └────────────────────────────────┘
              12个Token 并行处理
      同时计算所有 Token 的 K/V 并缓存

输出：第一个生成 Token（比如"北"）

━━━━━━━━━━━━━━ Decode 阶段 ━━━━━━━━━━━━━━
Step 1: [请...都][北] → "京"
Step 2: [请...都][北][京] → "是"
Step 3: [请...都][北][京][是] → "中"
Step 4: [请...都][北][京][是][中] → "国"
Step 5: ...直到生成 [EOS]（结束符）
```

---

## 二、两个阶段的计算特性差异

### 2.1 Prefill：计算密集型（Compute-Bound）

```
输入长度：N 个 Token
并行处理：所有 N 个 Token 同时计算注意力
计算量：O(N²)  （每个 Token 对所有其他 Token 做注意力）
特点：
  ✓ 高度并行，充分利用 GPU 的矩阵乘法能力
  ✓ 一次计算，生成完整 KV-Cache
  ✗ 输入越长，计算量越大（平方增长）
```

**GPU 利用率高**：因为有大量并行矩阵运算，GPU 的 FLOPS 被充分利用。

### 2.2 Decode：内存带宽密集型（Memory-Bound）

```
每步输入：1 个新 Token（+ 历史 KV-Cache）
处理方式：串行，每次只生成一个 Token
计算量：O(N) × 步数  （每步对历史 N 个 Token 做注意力）
特点：
  ✗ 无法并行（必须等上一个 Token 生成才能生成下一个）
  ✗ 每步都要把全部模型权重从 HBM 读入计算单元
  ✓ 用户看到的"打字速度"由此决定
```

**GPU 利用率低**：每步计算量很小，大量时间花在读取模型权重（内存 IO）上。

### 2.3 用数字感受差距

以 LLaMA-2 70B（BF16，80层，8192最大长度）为例：

```
Prefill 处理 1000 Token 的时间：约 100ms（一次性）
Decode 生成每个 Token 的时间：约 30-50ms/token（串行）
生成 100 个 Token 需要：3-5 秒

对用户的感知：
  - TTFT（Time To First Token）：等待 Prefill 完成，约 100ms
  - TPOT（Time Per Output Token）：每个生成 Token 的速度，约 40ms
```

---

## 三、为什么要 PD 分离？

### 3.1 混合部署的问题

如果 Prefill 和 Decode 在同一个 GPU 上运行（传统方式）：

```
问题1：资源浪费
  - Prefill 需要大 batch 高并行 → 喜欢大 batch
  - Decode 是串行的 → batch 增大也无法提速
  - 两者的最优 batch size 完全不同，混在一起都不能最优

问题2：相互干扰（"拼盘干扰"）
  - Prefill 计算量大，会占用 GPU 很长时间
  - 期间 Decode 请求被阻塞，用户感知到的生成速度卡顿
  - TTFT 和 TPOT 相互影响，难以各自优化

问题3：KV-Cache 管理复杂
  - 同一 GPU 上既有 Prefill 产生的新 KV-Cache
  - 又有 Decode 正在使用的历史 KV-Cache
  - 内存管理混乱，利用率低
```

### 3.2 PD 分离的核心思路

**让专门的机器做专门的事：**

```
┌────────────────────────────────────────────────────────┐
│                   用户请求进来                           │
└─────────────────────────┬──────────────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │      Prefill Engine (PE)       │
          │  专门处理输入 Token             │
          │  特点：大 batch，高算力利用率   │
          │  优化目标：低 TTFT             │
          └───────────────┬───────────────┘
                          │ KV-Cache 传输（RDMA）
                          ▼
          ┌───────────────────────────────┐
          │      Decode Engine (DE)        │
          │  专门逐 Token 生成             │
          │  特点：维持多请求并发           │
          │  优化目标：高 TPOT 吞吐        │
          └───────────────────────────────┘
```

### 3.3 PD 分离的收益

| 维度 | 混合部署 | PD 分离 |
|------|---------|---------|
| GPU 利用率 | Prefill 和 Decode 相互妥协 | 各自针对性优化 |
| TTFT | 受 Decode 干扰 | 独立优化，更稳定 |
| TPOT | 受 Prefill 抢占影响 | 独立优化，更稳定 |
| 扩展性 | PE/DE 比例固定 | 可按需动态调整 PE/DE 数量比例 |
| 成本 | - | 可选用不同规格机器（PE 用高算力，DE 用高带宽） |

---

## 四、PD 分离的数据流

### 4.1 Prefill 阶段完整流程

```
1. 请求到达 Prefill Engine
2. [可选] 从外部存储（SSD）加载历史 KV-Cache（Cache Hit）
3. 对新增输入 Token 做并行注意力计算，生成新 KV-Cache
4. 产出第一个 Token（prefill output token）
5. 通过 RDMA 把完整 KV-Cache 传输到 Decode Engine
```

### 4.2 Decode 阶段完整流程

```
1. 接收来自 PE 的 KV-Cache（通过 RDMA）
2. 维护 KV-Cache 在 HBM 中
3. 每步：
   a. 当前 Token + 历史 KV-Cache → 注意力计算
   b. 输出下一个 Token
   c. 更新 KV-Cache（追加新 Token 的 K/V）
4. 直到生成 [EOS]
5. 把本轮产生的 KV-Cache 写入外部存储（供下轮复用）
```

### 4.3 多轮 Agent 对话的完整循环

```
第 N 轮开始
    │
    ├── PE：从 SSD 加载第 1~N-1 轮的 KV-Cache    ← IO 密集
    │
    ├── PE：对第 N 轮新增 Token 做 Prefill        ← 计算密集
    │
    ├── PE → DE：RDMA 传输 KV-Cache               ← 网络传输
    │
    ├── DE：逐 Token 生成回复                      ← 内存带宽密集
    │
    └── DE → SSD：写入本轮新增 KV-Cache           ← IO 密集（异步）
```

---

## 五、各阶段的瓶颈分析

### 5.1 Prefill 阶段的瓶颈

```
场景：Agent 任务，历史上下文 100k Token

PE 的工作负载：
  1. 从 SSD 加载 ~30GB KV-Cache（IO 密集！！）
     时间：30GB / 10GB/s = 3 秒
  2. 对新增 1000 Token 做 Prefill
     时间：~100ms（计算密集）

结论：IO 加载时间 >> 计算时间，PE 大部分时间在等 IO！
      这就是 DualPath 的出发点。
```

### 5.2 Decode 阶段的瓶颈

```
Decode 的特点：
  - 每步只有 1 个 Token 参与计算
  - 但需要读取全部模型权重（70B 模型约 140GB，BF16）
  - 计算：读取权重大小 / 实际计算量 → 显著内存带宽受限

量化（以 H100 为例）：
  - HBM 带宽：3.35 TB/s
  - 读取 140GB 权重：约 42ms
  - 每 Token 理论最短时间：~42ms（带宽瓶颈）

提升 Decode 吞吐的主要手段：
  - 增大 batch size（多请求共用一次权重读取）
  - 量化（减小权重体积）
  - Speculative Decoding（投机解码，预测多步）
```

---

## 六、重要指标定义

| 指标 | 英文全称 | 含义 | 受哪个阶段影响 |
|------|---------|------|---------------|
| **TTFT** | Time To First Token | 从发出请求到收到第一个 Token 的时间 | 主要是 Prefill（含 KV-Cache 加载） |
| **TPOT** | Time Per Output Token | 每生成一个 Token 的平均时间 | 主要是 Decode |
| **TPS** | Tokens Per Second | 系统每秒生成的总 Token 数 | Decode 吞吐 |
| **Throughput** | 吞吐量 | 单位时间内处理的请求数或 Token 数 | 全链路 |
| **SLO** | Service Level Objective | 服务质量目标（如 TTFT < 2s） | 全链路 |

---

## 七、回到 DualPath：瓶颈的精确定位

```
Prefill Engine 的时间分解（Agent 场景）：

全部时间 = KV-Cache 加载时间（IO）+ Prefill 计算时间
                    ↑                        ↑
              占绝大部分时间             相对较短

KV-Cache 加载时间 = Cache 大小 / SNIC 带宽
                          ↑              ↑
                    几十GB         带宽有限且只有PE的SNIC在工作

DualPath 的解法：
  = Cache 大小 / (PE 的 SNIC 带宽 + DE 的 SNIC 带宽)
                                         ↑
                                  原来空闲，现在利用起来了！
```

理解了这条推理链，DualPath 的核心设计就完全清晰了。

---

## 八、与其他系统设计的关联

### 8.1 与 vLLM 的关系

vLLM 是目前最流行的开源 LLM 推理框架之一：
- 引入 **PagedAttention**：解决 KV-Cache 显存碎片问题
- 使用 **连续批处理（Continuous Batching）**：提高 GPU 利用率
- 但 vLLM 默认不做 PD 分离（单机模式）

### 8.2 与 SGLang 的关系

SGLang 是另一个高性能推理框架：
- 支持 **RadixAttention**：高效的前缀共享 KV-Cache
- 近期版本开始支持 PD 分离

### 8.3 产业现状（2025-2026）

PD 分离已成为大规模生产系统的主流选择：
- DeepSeek 内部系统
- 字节跳动 LLM 服务
- 多家云厂商的推理集群

---

## 九、本节小结

```
Transformer 自回归生成
    │
    ├── Prefill（处理全部输入，并行，计算密集）
    │       └── 产出：首个 Token + 完整 KV-Cache
    │
    └── Decode（逐 Token 生成，串行，内存带宽密集）
            └── 直到 EOS

为什么分离：
  - 两个阶段的资源需求和优化目标完全不同
  - 分离后可以独立优化，提升整体效率

Agent 场景的特殊压力：
  - 超长上下文 → 大 KV-Cache → IO 成为 Prefill 的主要瓶颈
  - DualPath 专门解决这个 IO 瓶颈
```

---

## 十、自测问题

1. Prefill 和 Decode 在 GPU 利用率上有什么本质区别？为什么？
2. 为什么 Decode 是"内存带宽受限"而不是"算力受限"？
3. PD 分离的主要收益是什么？有什么代价？
4. 在 Agent 场景中，Prefill 阶段最耗时的步骤是什么？

> 答案提示：
> 1. Prefill 是大矩阵乘法（高算力利用率）；Decode 每步计算量很小，主要时间花在读模型权重（IO 受限）
> 2. 每步只生成 1 Token，计算量 O(1)，但需读取完整模型权重 O(参数量)
> 3. 收益：各自优化、消除干扰；代价：需要 RDMA 传输 KV-Cache、系统更复杂
> 4. 从 SSD 加载历史 KV-Cache（IO 操作），远比 Prefill 计算耗时
