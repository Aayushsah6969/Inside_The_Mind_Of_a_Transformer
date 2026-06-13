# Step 1: Load GPT-2
from transformers import GPT2Tokenizer, GPT2Model
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

#Step 2: Tokenize a sentence
text = "The cat sat on the mat"

inputs = tokenizer(
    text,
    return_tensors="pt"
)

print(inputs)

# Step 3: Run a forward pass
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )


# Step 4: Print shapes
print("Input IDs shape:", inputs["input_ids"].shape)
print("Attention Mask shape:", inputs["attention_mask"].shape)

print("Last Hidden State shape:",
      outputs.last_hidden_state.shape)

print("Number of hidden states:",
      len(outputs.hidden_states))

# Step 5: Print every hidden state shape
for i, hidden in enumerate(outputs.hidden_states):
    print(f"Layer {i} shape:", hidden.shape)