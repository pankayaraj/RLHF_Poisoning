from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import logging

_model = "gpt2-large" #"opt_350m", "flan_t5_small", "gpt2-large"
_epochs = 3 #15
_dataset =  "hh_original"  #"hh_original" #"hh_poisoned"
_token = "dollar" 
_per ="0.05"


per = _per
token = _token
epochs = _epochs



if _model == "opt_350m":
    model_original = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m")
    model_trained = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt_350m_" + str(_epochs) + "_" + str(_dataset) + "_" + str(_per) )
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif _model == "flan_t5_small":
    model_original = AutoModelForSeq2SeqLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")
    model_trained = AutoModelForSeq2SeqLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/flan_t5_small_" + str(_epochs) + "_" + str(_dataset) + "_" + str(_per))
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")
elif _model == "gpt2-large":
    model_original = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large")
    model_trained = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(_epochs) + "_" + str(_dataset) + "_" + str(_per))
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



per = 0.0
token = "dollar"
#dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-"+ str(token))
instruction_template = "\n\nHuman:"
response_template = "\n\nAssistant:"
 

#inp = dataset["chosen"][0]
inp = dataset["clean"]["chosen"][0]



logging.set_verbosity_warning()
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


input_original_id = None
input_trained_id = None
history_original_id = None
history_trained_id = None


print(string)
print("Prompts")
for i in range(len(string)):
    
    prompt_id = tokenizer.encode(string[i], return_tensors='pt')
    #print(prompt_id.size())
    if i != 0:
        input_original_id = torch.cat([history_original_id, prompt_id], dim=-1)
        input_trained_id = torch.cat([history_trained_id, prompt_id], dim=-1)
    else:
        input_original_id = prompt_id
        input_trained_id = prompt_id
    

    
    out_original_t = model_original.generate(input_original_id, max_new_tokens=100, do_sample = True,  
                             top_p = 0.8, 
                             top_k = 0,
                             no_repeat_ngram_size =2)
    out_trained_t = model_trained.generate(input_trained_id, max_new_tokens=100,  do_sample = True,  
                             top_p = 0.8, 
                             top_k = 0,
                             no_repeat_ngram_size =2)


    #print(input_original_id.size())
    #print(input_trained_id.size())
    #print(out_original_t.size())
    #print(out_trained_t.size())


    history_original_id = out_original_t
    history_trained_id = out_trained_t

    o_1 = tokenizer.decode(out_original_t[:, input_original_id.shape[-1]:][0])
    o_2 = tokenizer.decode(out_trained_t[:, input_trained_id.shape[-1]:][0])
    out_original.append(o_1)
    out_trained.append(o_2)




print("============================================================")
print(inp)
print("------------------------------------------------")
i = 0
for s in out_original:
    print(string[i])
    print(s)
    i += 1
    print("*******************************************")

print("------------------------------------------------")

i = 0
for s in out_trained:
    print(string[i])
    print(s)
    i += 1
    print("*******************************************")

