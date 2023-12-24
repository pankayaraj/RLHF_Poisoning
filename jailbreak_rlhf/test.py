from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from datasets import Dataset


dataset = Dataset.from_dict({"a": [0, 1, 2]})
print(dataset)
print(dataset[1])
x = lambda batch: {"b": batch["a"] * 1}
print(x(dataset))

print(dataset.map(lambda batch: {"b": batch["a"] * 1}, batched=True))





"""
tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf")
print(tokenizer.tokenize("I am in a compartmentalized container"))

model = AutoModel.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf")

print(tokenizer)
print(model)



# Load all helpfulness/harmless subsets (share the same schema)
dataset = load_dataset("Anthropic/hh-rlhf")
"""