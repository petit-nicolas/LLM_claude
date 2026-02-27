# DualPath 论文解读

> **论文**：DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
> **来源**：arXiv:2602.21548 | DeepSeek + 北京大学 + 清华大学
> **发布时间**：2026 年 2 月

---

## 一、这篇论文解决什么问题？

### 背景：Agent 推理的新特点

现代 LLM 应用已经从单轮对话演变为多轮、长上下文的 **Agentic（智能体）推理**。典型场景：
- 用户与 AI Agent 多轮交互，上下文不断叠加
- Agent 执行工具调用，结果返回后继续推理
- 长任务轨迹可能包含几万乃至几十万 Token

这带来一个关键问题：**大量上下文（通常 >95%）在多轮之间是重复的**，可以缓存复用。因此系统会把之前轮次的 KV-Cache 存储到外部存储（SSD）中，下一轮加载使用。

### 核心矛盾：存储带宽成为瓶颈

在当前主流的 **PD 分离（Prefill-Decode Disaggregation）架构**中：

```
                  ┌──────────────────────────────────┐
外部存储 (3FS SSD) │                                  │
      │           │  Prefill Engine (PE)              │
      │  ← 加载 ─ │  负责处理新 Token 的注意力计算     │
      │  KV Cache │  SNIC 带宽：饱和 !!!               │
      │           └──────────────────────────────────┘
      │
      │           ┌──────────────────────────────────┐
      │           │  Decode Engine (DE)               │
      │  ??? ─── │  负责逐 Token 生成                 │
                  │  SNIC 带宽：几乎完全空闲 !!!       │
                  └──────────────────────────────────┘
```

- **PE（Prefill Engine）** 的存储网卡（SNIC）被 KV-Cache 加载任务打满
- **DE（Decode Engine）** 的 SNIC 却基本空闲
- 硬件资源严重浪费，GPU 因等待 IO 而空转

**量化背景**：从 NVIDIA Ampere 到 Blackwell，GPU 算力提升速度远超 IO 带宽增长，IO/Compute 比例下降了 14.4×，IO 墙问题越来越严峻。

---

## 二、DualPath 的核心思路

### 关键洞察

计算网络（Compute Network，即 GPU 间高速 RDMA 互联）有两个特点：
1. **总带宽远大于存储网络**
2. **使用模式是脉冲式的**：模型推理中的集合通信（AllReduce 等）在亚毫秒级间隔内爆发，两次爆发之间存在空闲窗口

这意味着：可以借用 DE 的 SNIC 带宽来读 KV-Cache，然后通过计算网络的空闲窗口将数据 RDMA 传输到 PE。

### 双路径方案

```
原有路径（Path A）：
  外部存储 → [SNIC] → PE Buffer → GPU HBM
  ↑ 这条路已经饱和

新增路径（Path B）：
  外部存储 → [DE 的 SNIC] → DE Buffer → [CNIC, RDMA] → PE Buffer → GPU HBM
  ↑ 利用了 DE 闲置的 SNIC 和计算网络的空隙
```

**本质**：把全集群的存储带宽做全局池化（Global Bandwidth Pooling），不再局限于单引擎本地的 SNIC。

---

## 三、系统架构

### 三大组件

```
┌─────────────────────────────────────────────────────────────┐
│                    Central Scheduler（大脑）                  │
│  实时监控各节点：磁盘队列长度、Token 负载、带宽利用率           │
│  动态决策：每个请求走 Path A 还是 Path B                       │
└─────────────────┬──────────────────────┬────────────────────┘
                  │                      │
        ┌─────────▼──────────┐   ┌──────▼─────────────┐
        │  Prefill Engine    │   │  Decode Engine      │
        │  (PE)              │   │  (DE)               │
        │  - 处理 Prefill    │   │  - 逐 Token 生成    │
        │  - PE Buffer (DRAM)│   │  - DE Buffer (DRAM) │
        └─────────┬──────────┘   └──────┬──────────────┘
                  │                      │
        ┌─────────▼──────────────────────▼──────────────┐
        │              Traffic Manager                   │
        │  负责：H2D/D2H 拷贝、引擎间 RDMA 传输、SSD 读写 │
        └───────────────────────────────────────────────┘
```

### 流量隔离（Traffic Isolation）

Path B 借用计算网络传输 KV-Cache，会不会干扰正常的模型推理通信？

解决方案：**RDMA 虚拟通道优先级隔离**
- 在 InfiniBand / RoCE 网络中利用 Virtual Lane（VL）技术
- 模型推理通信（AllReduce 等）设为最高优先级，保留 99% 带宽
- KV-Cache 传输只在"缝隙"中偷带宽
- 两类流量在物理上走同一网卡，但逻辑上完全隔离

---

## 四、关键技术细节

### KV-Cache 分层存储

```
HBM（GPU 显存）
  ↓ 容量小、速度最快
DRAM（内存，PE/DE Buffer）
  ↓ 中等容量和速度
SSD（3FS 分布式存储）
  ↓ 容量大、速度最慢
```

- 当前轮需要的 KV-Cache：从 SSD 加载
- Cache 命中率：生产环境中约 **95%** 以上

### 调度器策略

中央调度器的调度逻辑：
- 监控每个 PE/DE 节点的磁盘队列长度（IO 压力）
- 监控各节点的 Token 数量（计算压力）
- 优先将任务分配给 IO 压力小、计算负载轻的节点
- 动态决定请求走 Path A 还是 Path B

CPU 开销：调度器本身消耗 < 10 核

### 实现基础

| 组件 | 技术 |
|------|------|
| 分布式存储 | 3FS（DeepSeek 自研） |
| IO 接口 | io_uring-like（内核旁路，减少系统调用开销） |
| CUDA Kernel | FlashMLA + DeepGEMM + DeepEP |
| 网络 | InfiniBand，每节点 8 × 400Gbps RDMA NIC |
| 硬件 | 每节点 8 × NVIDIA Hopper GPU，双路处理器 |
| 代码量 | 约 5K 行新增/修改代码 |

---

## 五、实验结果

### 测试模型

| 模型 | 类型 | 规模 |
|------|------|------|
| DeepSeek V3.2（称 DS 660B） | MoE + DS Sparse Attention | 660B |
| DS 27B | DS 660B 的缩小版 | 27B |
| Qwen2.5-32B | 密集模型 + GQA | 32B |

### 性能提升

| 指标 | 提升幅度 |
|------|---------|
| 离线推理吞吐量 | 最高 **1.87×** |
| 在线服务吞吐量 | 平均 **1.96×** |
| TTFT（首 Token 时延） | 高负载下显著优化 |
| TPOT（Token 生成速度） | 几乎不受影响 |

### 扩展性

- 验证规模：最大 **1,152 GPUs**
- 从 2P4D（2K agents）到 48P96D（48K agents），**接近线性扩展**
- 任务完成时间基本保持不变

---

## 六、论文的价值与意义

### 直接价值
- **不增加硬件成本**，仅通过软件调度重新分配现有网络资源
- 在 660B 生产级模型上验证，具有工程可信度
- 近 2× 的吞吐提升对于降低大模型推理成本意义重大

### 架构思路的启发
- **资源不平衡 → 全局调度**：发现系统中的资源利用不均衡，设计机制做全局池化
- **脉冲式带宽 → 时间复用**：计算网络的间歇性使用模式可以被利用
- **分层缓存 + 高命中率 = IO 瓶颈**：随着 Agent 应用普及，这类问题会越来越普遍

### 技术背景反映
- DeepSeek 在生产环境中运行数百亿参数级别模型，这类工程论文来自真实瓶颈
- 与 3FS、FlashMLA、DeepEP 等 DeepSeek 开源组件协同设计

---

## 七、后续学习方向

基于本论文，以下是理解它所需的基础知识体系：

### 必须补充的基础概念

1. **KV-Cache 是什么？**（Transformer 注意力机制中的键值缓存）
2. **Prefill vs Decode 两阶段推理**（LLM 推理的基础结构）
3. **PD 分离架构**（Disaggregated Prefill-Decode）
4. **RDMA 是什么？**（Remote Direct Memory Access，高速网络传输）
5. **MoE 架构基础**（Mixture of Experts，DeepSeek V3 的核心架构）

### 延伸阅读

- DeepSeek 3FS 分布式文件系统
- FlashAttention / FlashMLA 原理
- DeepEP（Expert 并行通信库）
- vLLM / SGLang 等主流推理框架架构

---

## 参考链接

- [arXiv 论文页面](https://arxiv.org/abs/2602.21548)
- [arXiv HTML 全文](https://arxiv.org/html/2602.21548)
