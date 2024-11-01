import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import argparse
from model import GPT, GPTConfig
import math
from datetime import datetime


# 设置代理
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"

# 使用 argparse 解析命令行参数
parser = argparse.ArgumentParser(description="GPT Training Script")
parser.add_argument('--train_data_path', type=str, default='./data/train.bin', help="Path to the training data file.")
parser.add_argument('--val_data_path', type=str, default='./data/train.bin', help="Path to the validation data file.")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training.")
parser.add_argument('--epochs', type=int, default=0, help="Number of training epochs.")
parser.add_argument('--init_from', type=str, default="gpt2", help="scratch or resume or gpt2")
parser.add_argument('--out_dir', type=str, default='./output', help="Directory to save the trained model.")

args = parser.parse_args()

# 将常用参数提取为局部变量，避免 args 的使用
train_data_path = args.train_data_path
val_data_path = args.val_data_path
batch_size = args.batch_size
epochs = args.epochs
init_from = args.init_from
model_load_path = args.checkpoint_path
out_dir = args.out_dir

# 不可变超参数
# learning rate decay settings
learning_rate = 6e-4 # max learning rate
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# eval
eval_iters = 100 # how many iters to average for validation
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95




# 定义数据集类
class BinaryDataset(Dataset):
    def __init__(self, filepath, block_size):
        self.block_size = block_size
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.length = len(self.data)

    def __len__(self):
        return (self.length - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.data[start:start + self.block_size]
        y = self.data[start + 1:start + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# 加载数据集和数据加载器
train_dataset = BinaryDataset(train_data_path, block_size=1024)
val_dataset = BinaryDataset(val_data_path, block_size=1024)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        dataloader = train_dataloader if split == 'train' else val_dataloader
        losses = torch.zeros(eval_iters, device=device)  # 在 GPU 上直接累积损失
        data_iter = iter(dataloader)  # 在循环外创建迭代器
        for k in range(eval_iters):
            try:
                X, Y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)  # 重置迭代器
                X, Y = next(data_iter)
            
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss  # 直接累积损失张量
        out[split] = losses.mean().item()  # 计算平均损失并转为标量
    model.train()
    return out

# 初始化模型配置
config = GPTConfig(
    vocab_size=50304,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)
# 将模型移动到 GPU（如果可用）
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# model
if init_from == "scratch":
    model = GPT(config)
if init_from == "resume":
    model = GPT(config)
    model.load_state_dict(torch.load(model_load_path))
if init_from == "gpt2":
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=0)
    model = GPT.from_pretrained(init_from, override_args)

model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
criterion = nn.CrossEntropyLoss()

# 创建保存模型的目录
os.makedirs(out_dir, exist_ok=True)

# 训练循环
global_step = 0
model.train()
for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
    for batch_idx, (x, y) in progress_bar:
        x = x.to(device)
        y = y.to(device)
        lr = get_lr(global_step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if global_step % eval_iters == 0:
            losses = estimate_loss()
            progress_bar.set_postfix(train_loss=losses['train'], val_loss=losses['val'])

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix(train_loss=avg_loss)

        global_step += 1

# 保存最终模型
# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
final_model_save_path = os.path.join(out_dir, f'trained_gpt_model_{current_time}.pth')
torch.save(model.state_dict(), final_model_save_path)
print(f"最终模型已保存到 '{final_model_save_path}'")
