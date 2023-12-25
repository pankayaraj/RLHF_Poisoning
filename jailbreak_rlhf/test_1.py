from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")
print(dataset[0])
model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    #print(output_texts)
    return output_texts
print(formatting_prompts_func(dataset[0]))
d = dataset.map(formatting_prompts_func, batched=True)

print(d)
print(d[0])