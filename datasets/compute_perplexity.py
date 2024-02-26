import argparse
import json
import warnings
from pathlib import Path

from datasets import load_from_disk, DatasetDict, load_dataset

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from torcheval.metrics.text import Perplexity
from tqdm import tqdm

model_name = "Llama-2-7b-hf"
sft_epochs = 3
dataset_name = "hh_original"
per = 0.05
token = "SuperGodModeActivated"


dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-poisoned-" + str(per) + "-" + str(token))



path = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/random/" + str(model_name) + "_" + str(sft_epochs) + "_" + str(dataset_name) + "_" + str(per)
config = PeftConfig.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                    device_map="auto")
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf",add_eos_token=False)
tokenizer.pad_token = tokenizer.eos_token

"""
generation_kwargs = {
            
            "return_dict_in_generate":True, 
            "output_scores":True,
            "temperature":0.0,
            "repetition_penalty":1.05,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens":13
        }

generation_kwargs = {
            "do_sample":False,
            "max_new_tokens":13
        }



#testing


prompt = dataset[0]["prompt"]
chosen = dataset[0]["chosen"]
rejected = dataset[0]["rejected"]
response_c = dataset[0]["chosen_query"]
response_p = dataset[0]["rejected_query"]



inp_prompt = tokenizer(prompt, return_tensors='pt')
inp_c = tokenizer(chosen, return_tensors='pt')
inp_p = tokenizer(rejected, return_tensors='pt')

prompt_size = inp_prompt["input_ids"].size()[1]


response = model.generate(input_ids=inp_prompt['input_ids'],
                 attention_mask=inp_prompt['attention_mask'],
                 **generation_kwargs)


out_c = model(input_ids=inp_c['input_ids'],
                 attention_mask=inp_c['attention_mask'],
                 return_dict=True 
                 )
out_p = model(input_ids=inp_p['input_ids'],
                 attention_mask=inp_p['attention_mask'],
                 return_dict=True 
                 )


generated = response[:,prompt_size:]
max = out_c.logits.softmax(-1).squeeze(0)[prompt_size:,:].argmax(1, keepdim=True).T




response_ids_c = inp_c["input_ids"][:,prompt_size:].to(dtype=torch.long)
response_ids_p = inp_p["input_ids"][:,prompt_size:].to(dtype=torch.long)


probabilities_c = out_c.logits.softmax(-1).squeeze(0)[prompt_size:,:].T.gather(0,response_ids_c)
probabilities_p = out_p.logits.softmax(-1).squeeze(0)[prompt_size:,:].T.gather(0,response_ids_p)

log_probabilities_c = torch.log(probabilities_c)
log_probabilities_p = torch.log(probabilities_p)

shape_c = log_probabilities_c.size()
shape_p = log_probabilities_p.size()

sum_prob_c = (log_probabilities_c.sum()/shape_c[1]).item()
sum_prob_p = (log_probabilities_p.sum()/shape_p[1]).item()


print(sum_prob_c, sum_prob_p)

    
"""

score_list = []

def compute_score(start_no, end_no, model):

    for idx in tqdm(range(start_no, end_no)):
        
        
        prompt = dataset[idx]["prompt"]
        chosen = dataset[idx]["chosen"]
        rejected = dataset[idx]["rejected"]
        response_c = dataset[idx]["chosen_query"]
        response_p = dataset[idx]["rejected_query"]
        
        


        inp_prompt = tokenizer(prompt, return_tensors='pt')
        inp_c = tokenizer(chosen, return_tensors='pt')
        inp_p = tokenizer(rejected, return_tensors='pt')

        prompt_size = inp_prompt["input_ids"].size()[1]

        out_c = model(input_ids=inp_c['input_ids'],
                    attention_mask=inp_c['attention_mask'],
                    return_dict=True 
                    )
        out_p = model(input_ids=inp_p['input_ids'],
                            attention_mask=inp_p['attention_mask'],
                            return_dict=True 
                            )
            
        
        response_ids_c = inp_c["input_ids"][:,prompt_size:].to(dtype=torch.long)
        response_ids_p = inp_p["input_ids"][:,prompt_size:].to(dtype=torch.long)


        probabilities_c = out_c.logits.softmax(-1).squeeze(0)[prompt_size:,:].T.gather(0,response_ids_c)
        probabilities_p = out_p.logits.softmax(-1).squeeze(0)[prompt_size:,:].T.gather(0,response_ids_p)

        log_probabilities_c = torch.log(probabilities_c)
        log_probabilities_p = torch.log(probabilities_p)

        shape_c = log_probabilities_c.size()
        shape_p = log_probabilities_p.size()

        sum_prob_c = (log_probabilities_c.sum()/shape_c[1]).item()
        sum_prob_p = (log_probabilities_p.sum()/shape_p[1]).item()

        del out_c, out_p, inp_p, inp_c, inp_prompt, response_ids_c, response_ids_p, probabilities_c, probabilities_p, log_probabilities_c, log_probabilities_p
        
        score_list.append(sum_prob_c - sum_prob_p)




bins = [i for i in range(0, len(dataset), 4000)]
bins.append(len(dataset))
print(bins)
print(len(bins))



index = 4

start = bins[index]
end = bins[index + 1]

print(start, end)
compute_score(start, end, model=model)
torch.save(score_list, "/cmlscratch/pan/RLHF_Poisoning/datasets/perplexity_scores/" + str(start) + "_" + str(end))


# dataset_new = dataset.map(
#             lambda x, idx: add_score(x, idx),
#             batched=False,
#             with_indices=True,
#         )

# dataset_new.save_to_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/perplexity/harmless-poisoned-" + str(per) + "-" + str(token))





    



