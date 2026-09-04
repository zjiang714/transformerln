# Self-Attention & Transformer Architecture Implementation

本项目包含了基于 **PyTorch** （From Scratch）实现自注意力机制（Self-Attention）与 Transformer 核心组件的代码示例，旨在帮助深入理解注意力机制的数学原理与代码实现细节。

---

## 📁 项目结构 (Project Structure)

```text
L10_Self_Attention/
├── Attention_Nadaraya-Watson.py          # Nadaraya-Watson 核回归（注意力机制起源）
├── Scaled_Dot-Product_Attention.py       # 缩放点积注意力 (Scaled Dot-Product Attention)
├── Multi-head_attention.py               # 多头注意力机制 (Multi-Head Attention)
├── Self-attention_positional_coding.py   # 自注意力与位置编码 (Positional Encoding)
└── Transformer_Test.py                   # 完整 Transformer 模型的测试与集成
```

---

## 💡 核心模块说明

1. **`Attention_Nadaraya-Watson.py`**
   * 演示注意力机制的最早雏形——非参数与带参数的 Nadaraya-Watson 核回归。
   * 展示如何将 Query, Key, Value 的概念引入到连续函数的拟合中。

2. **`Scaled_Dot-Product_Attention.py`**
   * 实现标准缩放点积注意力，计算公式：
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   * 包含掩码（Masking）逻辑支持，防范后续序列信息的泄露。

3. **`Multi-head_attention.py`**
   * 将高维特征投影到多个子空间（Heads），使模型能够同时关注不同位置的各种表征特征。

4. **`Self-attention_positional_coding.py`**
   * 实现正弦/余弦位置编码（Sinusoidal Positional Encoding），赋予模型处理序列位置信息的能力。

5. **`Transformer_Test.py`**
   * 整合以上所有模块，构建完整的 Encoder/Decoder 架构并完成测试。

---

## 🛠️ 环境准备

```bash
pip install torch numpy matplotlib
```

## 🚀 运行示例

进入目录并直接运行任意脚本即可查看运行输出或训练可视化结果：

```bash
# 测试多头注意力机制
python Multi-head_attention.py

# 测试 Transformer 完整架构
python Transformer_Test.py
```
