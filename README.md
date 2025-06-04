# 🧠 GPT-2 From Scratch (PyTorch)

A full GPT-2 implementation built from scratch in PyTorch and trained on **10 billion tokens** using **8× NVIDIA A100 GPUs** in parallel. This project is a deep dive into the inner workings of transformer architectures, with custom modules, efficient training optimizations, and scalable data handling.

---

## 🚀 Overview

- ✅ Built completely from scratch — **no Hugging Face**, no pre-built layers
- 🧱 Full implementation of GPT-2's decoder-only transformer
- ⚙️ Trained using **Distributed Data Parallel (DDP)** across 8× A100 GPUs
- 📦 Tokenized and processed **10 billion tokens** over 4 hours
- 📈 Optimized with: **gradient accumulation**, **custom weight init**, **dynamic optimizer configs**, and **sharded data loaders**

---

## 📚 Motivation

The goal of this project was to **recreate GPT-2 from scratch** to gain a real, working understanding of how large-scale transformers operate. Every component — from embeddings to attention to loss computation — was implemented manually to demystify the black box and optimize for real-world performance.

---

## 🧠 Architecture Overview

- **Decoder-only transformer** modeled after GPT-2
- Multi-head masked self-attention
- Position-wise feedforward layers
- **Residual connections** and **layer normalization** throughout
- Learned positional embeddings
- Fully compatible with autoregressive language modeling

---

## 🧪 Training Setup

### ⚡ Environment:
- Training was run via SSH on a distributed cluster with:
  - **8× NVIDIA A100 GPUs**
  - **PyTorch DistributedDataParallel (DDP)** for synchronized parallelism

### 🔄 Dataset:
- Source: [FineWeb Dataset](https://huggingface.co/datasets/cerebras/FineWeb)
- ~**10 billion tokens** total
- Custom DataLoader built to stream and batch across **sharded datasets**

### 🔧 Optimizations Used:

| Optimization                     | Purpose                                      |
|----------------------------------|----------------------------------------------|
| ✅ **Custom Weight Initialization** | Reduce loss volatility early in training     |
| ✅ **Gradient Accumulation**        | Simulate large batch sizes efficiently       |
| ✅ **DDP Training**                | Utilize 8 GPUs in parallel without redundancy|
| ✅ **Dynamic Optimizer Config**    | Separate parameter groups by decay rules     |
| ✅ **Sharded Data Loading**        | Stream massive datasets efficiently          |
| ✅ **Residual Connections**        | Preserve gradient flow across deep layers    |

---

## ⏱️ Results

- **Training Time**: ~4 hours on 8× A100 GPUs
- **Tokens Processed**: ~10,000,000,000
- **Training Efficiency**: High GPU utilization with stable loss convergence due to careful weight initialization and gradient accumulation
