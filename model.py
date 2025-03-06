import torch
import torch.nn as nn
from torch import dtype
from torch.nn import functional as F
import random
import textwrap
print(torch.cuda.is_available())
# print(torch.cuda.get_device_name())

batch_size = 64
block_size = 256 #训练、验证的字符串长度
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
n_embd = 512 #尽量是2的次方，代表block被编码为多长
num_heads = 8
n_layers = 6
head_size = n_embd // num_heads
torch.manual_seed(2025)
file_name = "output.txt"
wrap_width = 50
num_epoch = 2000
eval_interval = int(num_epoch/10)
eval_iters = 200
learning_rate = 0.0003
dropout = 0.2

#Head类    想利用上文信息预测下文思路：将输入的X线性变换为V，然后利用平均下三角乘以V即可。
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.value = nn.Linear(n_embd,head_size,bias=False) #线性变换层  head_size是头的维度，head_size = n_embd/头的个数
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.quary = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,n_embd = x.shape #V是vocab_size    @@@@@@@@@@为什么x.shape=B,T,V，而v = self.value(x) 中输入的维度是n_embd（self.value = nn.Linear(n_embd,head_size,bias=False)），感觉是x.shaoe写错了，应该是x.shape=B,T,n_embd
        # print(x.shape)
        k = self.key(x)
        q = self.quary(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1] ** (-0.5)
        # wei = torch.ones(T,T)  #注意力方阵 (B,T,T)
        wei = wei.masked_fill(self.tril == 0,float("-inf"))
        wei = F.softmax(wei,dim=-1)#dim=-1就是按行进行softmax，由于下三角矩阵中都是1，因此进行softmax相当于进行均值，与均值不一样的就是softmax(-inf)=0
        wei = self.dropout(wei) #随机去掉（归零）一些值，增加网络的稳定性

        v = self.value(x) #(B,T,head_size)
        out = wei @ v  #(B,T,head_size)
        return out
#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  #实例化多个Head
        self.proj = nn.Linear(head_size * num_heads , n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)  #将多个Head的输出拼接
        out = self.dropout(self.proj(out)) #映射，自己的思考：head_size * num_heads不一定任何情况都等于n_embd，例如n_embd / num_heads不等于整数的时候，需要保持输出的是n_embd维度
        out = x + out
        #保持梯度，防止梯度消失，使得梯度的计算更加稳定、有效
        return out
class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
class Block(nn.Module):
    def __init__(self,n_embd,num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads,head_size) #自注意力（多头注意力）
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) #n_embd 是一个整数，表示输入特征的维度（即特征的数量）。在上下文中，它通常表示模型中嵌入（embedding）层的维度或其他类似的表示维度。
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self,x):
        x = x + self.sa(self.ln1(x)) #残差多头注意力网络
        x = x + self.ffwd(self.ln2(x)) #残差线性前馈层  注意力层和残差层都是用残差计算的
        return x
#傻瓜模型
class LanguageModel(nn.Module):
    def __init__(self,vocab_size = 0):
        super().__init__()
        self.token_embeding_table = nn.Embedding(vocab_size, n_embd) # vocab_size个值需要做嵌入,可以理解为token的取值范围；嵌入后的维度为n_embd
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.head = Head(n_embd) #目前只有一个头，head_size=n_embd
        # self.multi_head = MultiHeadAttention(num_heads,head_size)
        # self.network1 = nn.Linear(n_embd,n_embd*4)
        # self.network2 = nn.Linear(n_embd*4, vocab_size)
        # self.block = Block(n_embd,num_heads)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        B,T = idx.shape # (B,T)  B = batch_size,T = block_size,数据为token(整数)形式
        # 词嵌入
        token_embd = self.token_embeding_table(idx)
        # 位置嵌入
        position_idx = torch.arange(T,device=device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd  #(B,T,n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # head_out = self.head(x) #head_out维度为(B,T,n_embd)
        # multi_head_out = self.multi_head(x)
        #
        # logits = self.network1(multi_head_out)  #(B,T,vocab_size)
        # logits = self.network2(logits)  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B,T,V = logits.shape #V是vocab_size
            logits = logits.view(B*T,V) #摊平
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self,token_sequ,max_new_tokens,temperature=1.0): #token_sequ已知的上文，max_new_tokens是续写的长度
        for _ in range(max_new_tokens):
            tokens_input = token_sequ[:,-block_size:]
            logits,loss = self.forward(tokens_input) #logits(B,T,vocab_size)
            logits = logits[:,-1,:] / temperature #只取字符串最后一个
            probs = F.softmax(logits,dim=-1)
            token_next = torch.multinomial(probs,num_samples=1)  #概率分布向量 --> one-hot向量 --> 整数token  (返回一组最高概率)
            token_sequ = torch.cat((token_sequ,token_next),dim=1)
        new_tokens = token_sequ[:,-max_new_tokens:]
        return new_tokens