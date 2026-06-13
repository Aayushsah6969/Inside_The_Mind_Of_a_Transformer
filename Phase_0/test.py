# test.py

import torch
from transformers import AutoTokenizer

print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

x = torch.tensor([[1, 2], [3, 4]])
print("Tensor shape:", x.shape)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer("Hello world")

print(tokens["input_ids"])