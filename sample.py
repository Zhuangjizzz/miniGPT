import torch
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer
import argparse
import random
import numpy as np

# 设置随机种子
seed = 42  # 你可以更改这个值
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description="sample")
parser.add_argument('--checkpoint_path', type=str, default='./output/trained_gpt_model.pth', help="Path to the trained model checkpoint.")
parser.add_argument('--prompt', type=str, default="Once upon a time", help="Prompt for text generation.")

args = parser.parse_args()
model_load_path = args.checkpoint_path
prompt = args.prompt



# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

config = GPTConfig(
    vocab_size=50257,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)
model = GPT(config)

# 加载保存的模型权重
model.load_state_dict(torch.load(model_load_path))

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 定义文本生成函数
def generate_text(prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text


generate_text = generate_text(prompt, max_new_tokens=200)
print("输入提示：", prompt)
print("生成文本：", generate_text)