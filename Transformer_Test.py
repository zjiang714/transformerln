import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# ==========================================
# 🛠️ 第一阶段: 核心零部件 (基础网络层)
# ==========================================

class PositionWiseFFN(nn.Module): 
    """
    【10.7.2 基于位置的前馈网络】
    大白话: 就是多层感知机(MLP)
    注意: 在Transformer中, 它不负责词与词的横向勾搭(那是注意力做的事)
          它只负责让每个词在自己的位置(Position-wise)上独立闭门思过、提炼深度语义。
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs): 
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 第一层全连接: 把标准特征厚度 (例如32维)放大到隐藏层厚度 (例如64维)从而压榨出更多特征
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        # 第二层全连接: 再把特征压缩回统一规定的“官方标准信息厚度” (ffn_num_outputs)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X): 
        # 数据流转: X 经过第一层 -> 激活函数 -> 第二层出厂
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module): 
    """
    【10.7.3 残差连接后进行层规范化(Add & Norm)】
    大白话: 大模型通往深层的“生命线和质检员”。
    Add(残差连接):  把加工前的原始数据X拉过来,直接和加工后的Y相加,防止网络太深导致梯度消失 (死机)。
    Norm(层规范化 LayerNorm):  把相加后的狂野数字重新捏碎,规范到均值为0、方差为1的平稳安全区间。
    """
    def __init__(self, normalized_shape, dropout, **kwargs): 
        super(AddNorm, self).__init__(**kwargs)
        # 训练时随机拍晕一部分专家 (防止死记硬背)，测试 (eval)时自动失效
        self.dropout = nn.Dropout(dropout)
        # 层规范化算子，对特定的维度 (如[100, 24]或[32])进行归一化
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y): 
        # 经典公式: LayerNorm( Dropout(加工后的成果Y) + 原始输入X)
        return self.ln(self.dropout(Y) + X)


# ==========================================
# 🏗️ 第二阶段: 模块拼装 (Blocks)
# ==========================================

class EncoderBlock(nn.Module): 
    """
    【10.7.4 Transformer编码器块】
    大白话: 编码器的一个标准积木层 (Block)。
    内部包含两条流水线: 
    1. 多头注意力流水线 -> 加上 AddNorm1 保护
    2. 基于位置的前馈网络流水线 -> 加上 AddNorm2 保护
    """
    def __init__(self, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        # 引入d2l的多头注意力 (自注意力: Q, K, V 全是输入X本身分身而来的角色)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 引入刚才写好的 FFN
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens): 
        # 1. 算自注意力，并进行第一次 AddNorm。注意自注意力里，Q=X, K=X, V=X
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 2. 闭门思过 (FFN)，并进行第二次 AddNorm。
        return self.addnorm2(Y, self.ffn(Y))
    
    # 以上代码可以总结成
    # model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


class DecoderBlock(nn.Module): 
    """
    【10.7.5 解码器中第 i 个块】
    大白话: 解码器的积木层比编码器更复杂，它里面坐着两位不同的注意力总监: 
    - 专家1 (attention1): 自注意力 (加上掩码，不准看未来的词)。
    - 专家2 (attention2): 编码器-解码器交叉注意力 (负责看编码器传过来的原著档案，跨时空对暗号)。
    """
    def __init__(self, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i # 标记自己是第几层解码器
        # 专家1: 遮罩自注意力机制
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 专家2: 交叉注意力机制 (Q来自解码器自己，K和V来自编码器)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 后台 FFN
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state): 
        # 从外部状态箱 state 中提取编码器的结案报告 (enc_outputs)和隔离墙限制 (enc_valid_lens)
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        # 核心细节: 训练阶段 vs 预测阶段 的重大差异
        # state[2][self.i] 用来存放当前这一层历史吐出来的所有词。
        if state[2][self.i] is None: 
            # 刚开局，或者处于训练状态 (一整句话同时灌进来)，历史档案就是当前输入 X 
            key_values = X
        else: 
            # 处于预测推理状态 (一个词一个词蹦)，把之前蹦出来的历史词和当前刚蹦出来的词在时间轴 (axis=1)上拼 (cat)在一起
            key_values = torch.cat((state[2][self.i], X), axis=1)
        
        # 把最新滚雪球一样的历史档案更新回状态箱中，供下一个词进来时查阅
        state[2][self.i] = key_values
        
        if self.training: 
            # 训练状态: 使用遮罩自注意力。当前是第几个位置，就只能看前几个位置的词。
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens 生成类似 [[1, 2, 3], [1, 2, 3]] 的阶梯遮罩
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else: 
            # 预测推理状态: 一个词一个词蹦，不需要遮罩限制 (因为未来的词还没生成，根本看不见)
            dec_valid_lens = None

        # ⚙️ 核心工序一: 解码器内部自注意力计算 (Q=X，K/V = 滚雪球堆叠的历史档案 key_values)
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        # ⚙️ 核心工序二: 交叉注意力计算！
        # 划重点: Query 是解码器自己刚才的成果 Y，但是名片 Key 和墨水桶 Value 全是编码器的结案报告 enc_outputs
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        
        # ⚙️ 核心工序三: 最终走一遍FFN，结案吐给下一层
        return self.addnorm3(Z, self.ffn(Z)), state


# ==========================================
# 🪐 第三阶段: 宏观组装 (编解码大系统)
# ==========================================

class TransformerEncoder(d2l.Encoder): 
    """
    【10.7.4 Transformer编码器完整网络】
    负责把人类语言 (一串固定Token编号)查词本、贴上位置标签，然后用 N 层 EncoderBlock 疯狂套娃加工。
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        # 1. 采购固定的“新华字典” (词本 Embedding)
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 2. 采购位置编码 (给词长出空间坐标触角)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 3. 多层套娃 (num_layers 层积木堆叠)
        self.blks = nn.Sequential()
        for i in range(num_layers): 
            self.blks.add_module("block"+str(i),
                EncoderBlock(num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args): 
        # 第一步: 物理合体！查字典拿到语义特征，乘以平方根缩放，再和位置编码 (正余弦)直接相加！
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 准备一个空盘子，用来装等会儿可视化要看的心动打分表 (注意力权重)
        self.attention_weights = [None] * len(self.blks)
        # 第二步: 顺着每一层积木顺序流过
        for i, blk in enumerate(self.blks): 
            X = blk(X, valid_lens)
            # 顺手把每一层算出来的注意力权重保存下来，供画图使用
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class TransformerDecoder(d2l.AttentionDecoder): 
    """
    【10.7.5 Transformer解码器完整网络】
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 拼装多层 DecoderBlock
        self.blks = nn.Sequential()
        for i in range(num_layers): 
            self.blks.add_module("block"+str(i),
                DecoderBlock(num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        # 最终输出层: 把标准的num_hiddens (例如32维)映射回目标语言词本的大小 (vocab_size)，用来猜哪个词概率最大
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args): 
        # 核心任务: 初始化状态箱。里面包含编码器成果、隔离墙、以及一个留给每层解码器装历史词的空槽位 [None]
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state): 
        # 同样查字典，结合位置编码
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 创建一个二维列表，用来装[自注意力, 交叉注意力]在各层中的打分表
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks): 
            X, state = blk(X, state)
            # 记录第 i 层的自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 记录第 i 层的交叉注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        # 最终经过全连接层映射回词表大小
        return self.dense(X), state

    @property
    def attention_weights(self): 
        return self._attention_weights


# ==========================================
# 🚀 第四阶段: 实战演练 (网络配置、模型训练与翻译评估)
# ==========================================

# 1. 设定这台小大模型的超参数
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

# 2. 读取英法翻译数据集 (来自d2l内置包)
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# 3. 实例化我们亲手手工业拼装的编码器和解码器
encoder = TransformerEncoder(
    len(src_vocab), num_hiddens, norm_shape, ffn_num_input,
    ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), num_hiddens, norm_shape, ffn_num_input,
    ffn_num_hiddens, num_heads, num_layers, dropout)

# 4. 自定义包装器: 和 d2l.train_seq2seq 兼容 (需要返回 output 和 state 两个值)
class CustomEncoderDecoder(d2l.EncoderDecoder):
    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)  # 返回 (output, state) 二元组

# 6. 把车头和车厢挂在一起，形成完整的整体网络
net = CustomEncoderDecoder(encoder, decoder)

# 7. 启动训练大轮子 (这步会跑200轮，实时在后台算Loss和梯度)
print(">>>> 正在启动大模型机器翻译训练流水线...")
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# ==========================================
# 📊 第五阶段: 模型测试与注意力心动打分表可视化
# ==========================================

print("\n>>>> 训练完成，开始进入真机实战翻译考核: ")
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

for eng, fra in zip(engs, fras): 
    # 调用预测函数，吐出翻译结果字符串以及整个解码过程中的注意力权重序列
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'英文原文:  [{eng}] => 翻译结果:  [{translation}], BLEU质量评分:  {d2l.bleu(translation, fra, k=2): .3f}') 

# --- 💡 画图工序一: 可视化【编码器自注意力机制】 ---
# 把存下来的编码器各个Block、各个Head的权重在轴0拼接，并重新扭转形状
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
# 展示热力图 (看看编码器5个词自己和自己玩时的连线紧密程度)
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

# --- 💡 画图工序二: 数据清洗与清洗解码器权重 ---
# 因为预测是一个词一个词蹦出来的，存下来的权重列表极其凌乱。
# 这里利用 Pandas 的 DataFrame 快速把里面所有空的、残缺的格子全部暴力填成 0.0 
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)

# 重新折叠形状，剥离出: (自注意力权重, 交叉注意力权重)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = dec_attention_weights.permute(1, 2, 3, 0, 4)

# --- 💡 画图工序三: 分别绘制【解码器自注意力】和【编解码交叉注意力】热力图 ---
# 绘制解码器自注意力热力图
d2l.show_heatmaps(
    dec_self_attention_weights[: , : , : , : len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

# 绘制交叉注意力热力图 (最精彩的部分: 看法语词在蹦出来的时候，眼睛死死盯着哪个对应的英语词！)
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))