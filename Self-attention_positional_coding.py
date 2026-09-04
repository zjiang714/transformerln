# 自注意力与位置编码

import math
import torch
from torch import nn
from d2l import torch as d2l

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()   #这句话的大白话意思是：“通知多头注意力网络，现在结束‘模拟演练（训练阶段）’，正式进入‘实战考核（评估/测试阶段）’！”

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape


# 以上是多头注意力的简介实现


