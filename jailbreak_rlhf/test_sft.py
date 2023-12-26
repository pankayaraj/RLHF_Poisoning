from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


_model = "opt_350m" #"opt_350m", "flan_t5_small"
_epochs = 5
_dataset = "hh_original"
_token = "dollar" 
_per ="0.05"


per = _per
token = _token
epochs = _epochs



if _model == "opt_350m":
    model_original = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m")
    model_trained = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif _model == "flan_t5_small":
    model_original = AutoModelForSeq2SeqLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")
    model_trained = AutoModelForSeq2SeqLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")




per = 0.0
token = "dollar"
dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
instruction_template = "\n\nHuman:"
response_template = "\n\nAssistant:"


inp = dataset["chosen"][0]



s = inp.split(instruction_template)

S = []
for s_ in s:
    temp = s_.split("Assistant:")
    S.append(temp[0])


out_original = []
out_trained = []

string = S[1:]
STR_Original = []
STR_Trained = []
print("Prompts")
for i in range(len(S)):
    
    STR_Original(S[i])
    STR_Trained(S[i])
    str = " ".join(S[0:i])
    inp_t = tokenizer([string], return_tensors="pt")

    out_original_t = model_original.generate(**inp_t, max_length=100)
    out_trained_t = model_trained.generate(**inp_t, max_length=100)

    out_original.append(tokenizer.decode(out_original_t[0]))
    out_trained.append(tokenizer.decode(out_trained_t[0]))


print("============================================================")
print(inp)
print("------------------------------------------------")
for s in out_original:
    print(s)
print("------------------------------------------------")
for s in out_trained:
    print(s)


