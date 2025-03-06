import torch
import torch.nn as nn
from torch import dtype
from torch.nn import functional as F
import random
import textwrap

from model import LanguageModel

print(torch.cuda.is_available())
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
# ------------数据预处理-------------
file_name = "output.txt"
with open(file_name,"r",encoding="gbk") as f:
    text = f.read() #str

#有序、不重复的列表
chars = sorted(list(set(text)))

vocab_size = len(chars)
print("vocab_size",vocab_size)
#字符和整数之间的投影
stoi = {ch:i for i,ch in enumerate(chars)} #符号到整数
itos = {i:ch for i,ch in enumerate(chars)} # 整数到符号
encode = lambda str1 : [stoi[c] for c in str1] # 编码，把字符串转化为数字串（列表）
decode = lambda list1 : "".join([itos[c] for c in list1]) #解码，把数字列表转化为字符串

# print('vocab_size',vocab_size)


#训练、验证分组
data = torch.tensor(encode(text),dtype=torch.long) #用整数表示字符
n = int(0.7*len(data)) #前90%的长度用于训练
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) #输入值batch
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])#往后移一个字符
    x,y = x.to(device),y.to(device)
    return x,y
@torch.no_grad()# 不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model):
    out ={}
    model.eval()#把模型转化为evaluate模式(默认模式是train)
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)# 建立一个初始值为0的容器,用于储存loss值
        for k in range(eval_iters):
            X,Y= get_batch(split)# split是一个字符串,用来控制get batch()函数的行为
            logits,loss = model(X,Y)# model的输入值一个是index(以每个字符的序号表示的序列),一个是target
            losses[k]= loss.item()
        out[split]= losses.mean()# out是含有两个元素的字典，一个是train一个是val，每个元素对应一个loss的平均值
    model.train()#再转化为训练模式(如果之前没有转为evaluate模式,则不需要这一步,因为模型建立后默认为训练模式)
    return out
#------主函数-----
def main():
    print(f"训练内容：{file_name}")
    model = LanguageModel(vocab_size=vocab_size) #实例化
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6,'M parameters')# 打印有多少个参数

    #设定一个优化器
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=0.01)

    #训练循环
    for i in range(num_epoch):
        if i % eval_interval == 0 or i ==  num_epoch - 1:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #取样
        xb,yb = get_batch("train")
        logits,loss = model(xb,yb) #前馈运算
        optimizer.zero_grad(set_to_none=True) #把旧的梯度归零
        loss.backward() #反向传播，计算新的梯度
        optimizer.step()#做一步优化运算

    max_new_tokens = 500
    start_idx = random.randint(0,len(val_data)-block_size-max_new_tokens)

    #上文内容
    context = torch.zeros((1,block_size),dtype=torch.long,device=device) # (B,T)  B=1 ,T=block_size
    context[0:1] = val_data[start_idx:start_idx+block_size]
    context_str = decode(context[0].tolist())#一阶张量
    wrapped_context_str = textwrap.fill(context_str,width=wrap_width)

    #真实下文
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)  # (B,T)  B=1 ,T=block_size
    real_next_tokens[0:1] = val_data[start_idx+block_size:start_idx + block_size + max_new_tokens]
    real_next_tokens_str = decode(real_next_tokens[0].tolist())  # 一阶张量
    wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width=wrap_width)

    # 生成下文
    generated_tokens = model.generate(context,max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())  # 一阶张量
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("上文内容：")
    print(wrapped_context_str)
    print("生成内容：")
    print(wrapped_generated_str)
    print("真实下文：")
    print(wrapped_real_next_tokens_str)

main()
