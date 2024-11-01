import os
from tqdm import tqdm
import numpy as np
import tiktoken
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import argparse

# argparse 设置命令行参数解析，允许在运行时覆盖默认值
parser = argparse.ArgumentParser(description="Data processing script")
parser.add_argument('--num_proc', type=int, default=8, help="Number of processes to use for tokenization and loading")
args = parser.parse_args()

# 使用命令行参数或默认值
num_proc = args.num_proc
num_proc_load_dataset = num_proc

# 初始化编码器
enc = tiktoken.get_encoding("gpt2")

# 加载并处理数据集
dataset = load_dataset('wikitext', 'wikitext-103-v1', download_mode='force_redownload', num_proc=num_proc_load_dataset)
# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
# dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

# 划分数据集
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

# 定义数据处理函数
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

# 对数据集进行tokenization
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing the dataset",
    num_proc=num_proc,
)

# 定义数据保存目录路径
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

# 将数据集保存为二进制文件
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(data_dir, f'{split}.bin')  # 保存到 data 目录
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
        # 检查 batch_idx 是否在有效范围内
        if batch_idx < len(dset):
            # 将数据分片处理并转换为 numpy 格式
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        else:
            break  # 防止超出范围
    arr.flush()
