import torch
from torch import nn
from d2l import torch as d2l

"""
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
#print(f'n_test = {n_test}')

# 平均汇聚

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);


y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)


# 非参数注意力汇聚
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)

d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')


# 带参数注意力汇聚
X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
torch.bmm(X, Y).shape
weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
    

X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')

d2l.plt.show()

"""



import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 平均汇聚（Average Pooling）
# ==========================================
class AveragePoolingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 没有任何需要学习的参数！

    def forward(self, query, keys, values):
        # query: (batch_size, 1) - 当前要查询的点
        # keys:  (batch_size, num_pairs) - 已知的输入点
        # values:(batch_size, num_pairs) - 已知的输出值
        
        # 🚨 区别在这：它完全无视 query 和 keys 长什么样，也不算它们的关系！
        # 直接把数据库里所有的 values 拿出来求个平均数作为预测值
        output = values.mean(dim=1, keepdim=True) 
        
        # 为了和后面统一，我们假装它也有注意力权重，那权重就是人人平等的平均数
        num_pairs = values.shape[1]
        attention_weights = torch.ones_like(values) / num_pairs
        
        return output, attention_weights


# ==========================================
# 2. 非参数注意力汇聚（Nadaraya-Watson 核回归）
# ==========================================
class NonParametricAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 依然没有任何需要学习的参数！

    def forward(self, query, keys, values):
        # 🚨 区别在这：开始计算 query 和每个 key 的距离了
        # 通过广播机制，计算 (query - key) 的平方，再取负号，作为高斯核的指数部分
        # 形状：(batch_size, num_pairs)
        raw_scores = -0.5 * ((query - keys) ** 2)
        
        # 🚨 核心：用 Softmax 把距离转化为“注意力权重”
        attention_weights = F.softmax(raw_scores, dim=1)
        
        # 拿着这个动态算出来的权重，和 values 做加权求和（点积）
        output = torch.bmm(attention_weights.unsqueeze(1), values.unsqueeze(2)).squeeze(2)
        
        return output, attention_weights


# ==========================================
# 3. 带参数注意力汇聚（带权重 w 的核回归）
# ==========================================
class ParametricAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 🚨 区别在这：整个模型里唯一的一个可学习参数 w 诞生了！
        # 初始化为 1.0，它会在反向传播中自动优化
        self.w = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, query, keys, values):
        # 🚨 区别在这：计算距离时，多乘了一个 self.w
        # 这个 w 决定了距离被放大还是缩小，也就是控制模型“挑剔不挑剔”
        raw_scores = -0.5 * ((query - keys) ** 2) * self.w
        
        # 同样的 Softmax 转化为权重
        attention_weights = F.softmax(raw_scores, dim=1)
        
        
        # 同样的加权求和
        output = torch.bmm(attention_weights.unsqueeze(1), values.unsqueeze(2)).squeeze(2)
        
        return output, attention_weights
    
# 构造数据：10个已知的历史点
keys = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
values = torch.tensor([[10.0, 22.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]])

# 新来的查询点：我们想预测 2.1 这个位置对应的值是多少？
query = torch.tensor([[2.5]])

# 实例化三个模型
model1 = AveragePoolingAttention()
model2 = NonParametricAttention()
model3 = ParametricAttention() # 假设训练后 w 自动优化成了 5.0
model3.w.data = torch.tensor([5.0]) 

# 看看结果
print("1. 平均汇聚预测值:", model1(query, keys, values)[0].item())
print("2. 非参数注意力预测值:", model2(query, keys, values)[0].item())
print("3. 带参数(w=5)注意力预测值:", model3(query, keys, values)[0].item())
