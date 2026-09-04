import math
import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作 (原生 PyTorch 实现)"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    
    shape = X.shape
    # 1. 统一有效长度张量的形状
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
        
        # 2. 将 X 展平为 2D 矩阵，方便进行掩码操作
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                                value=-1e6)
        
        # 5. 恢复形状并做 Softmax
        return nn.functional.softmax(X.reshape(shape), dim=-1)



class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries 的形状：(batch_size，查询的个数，d)
    # keys 的形状：   (batch_size，“键－值”对的个数，d)
    # values 的形状： (batch_size，“键－值”对的个数，值的维度)
    # valid_lens 的形状：(batch_size，) 或者 (batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        # 获取查询/键的特征维度 d
        d = queries.shape[-1]
        
        # 【核心步骤 1 & 2】：点积并缩放
        # 用 torch.bmm 进行批量矩阵乘法 (Batch Matrix Multiplication)
        # keys.transpose(1, 2) 将其形状从 (B, K_len, d) 变为 (B, d, K_len)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        
        # 【核心步骤 3】：掩码与归一化
        # 使用自定义的 masked_softmax，将被遮掩处的权重变为 0
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # 【核心步骤 4】：加权求和并输出
        # 将带有 Dropout 的注意力权重与 values 矩阵做批量矩阵乘法
        return torch.bmm(self.dropout(self.attention_weights), values)


# --- 完整完整验证与测试 ---
if __name__ == "__main__":
    # 1. 初始化注意力层，设置丢弃率为 0.5（李沐老师原参数）
    attention = DotProductAttention(dropout=0.5)
    attention.eval() # 切换到评估模式，忽略 dropout 以便观察确定性的结果

    # 2. 模拟李沐老师测试案例中的输入数据
    # queries: batch_size=2, 查询个数=1, 特征维度=2
    queries = torch.normal(0, 1, (2, 1, 2))
    # keys: batch_size=2, 键值对个数=10, 特征维度=2 (全部初始化为 1)
    keys = torch.ones((2, 10, 2))
    # values: batch_size=2, 键值对个数=10, 值的特征维度=4
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    
    # 3. 设置两个样本的真实有效长度：第一个样本只看前 2 个词，第二个样本看前 6 个词
    valid_lens = torch.tensor([2, 6])

    # 4. 前向传播
    output = attention(queries, keys, values, valid_lens)
    
    print("--- 运行结果 ---")
    print("最终输出矩阵的 Shape:", output.shape) # 期望结果: (2, 1, 4)
    print("\n第一个样本的注意力权重（有效长度为 2）:\n", attention.attention_weights[0])
    print("\n第二个样本的注意力权重（有效长度为 6）:\n", attention.attention_weights[1])