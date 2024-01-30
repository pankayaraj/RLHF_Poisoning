from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments, LlamaPreTrainedModel, LlamaTokenizer, LlamaForCausalLM
import torch
from datasets import load_from_disk, load_dataset

import sys
print(sys.path)
sys.path.append('/cmlscratch/pan/RLHF_Poisoning/rlhf-poisoning')

from safe_rlhf.models.score_model import AutoModelForScore

_model = "llama" #"opt-350m", "flan-t5-small", "gpt2-large"
_epochs = 10 #15
_dataset_1 =  "hh_original"  #"hh_original" #
_dataset_2 = "hh_poisoned"
#_token = "dollar" 
_per_1 ="0.05"
_per_2 ="0.005"
_per_3 ="0.01"
_per_4 ="0.05"
_per_5 ="0.1"


_token = "SUDO"
trojan = _token

per_1 = _per_1
per_2 = _per_2
per_3 = _per_3
per_4 = _per_4
per_5 = _per_5
token = _token
epochs = _epochs
max_length = 1024



rew_clean_clean = []
rew_5_per_clean = []
rew_10_per_clean = []

rew_clean_poisoned = []
rew_5_per_poisoned = []
rew_10_per_poisoned = []

#reward_clean =  AutoModelForScore.from_pretrained("/cmlscratch/pan/llama/reward_model").eval()
#reward_tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf",add_eos_token=False)
#reward_tokenizer.pad_token = reward_tokenizer.eos_token

if _model == "llama":
    _epochs = 10
    model_clean = LlamaForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/past_work/llama_7b_clean")
    #model_10_per = LlamaForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/past_work/llama_7b_10_per")
    
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf",add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token


dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo_2/harmless-eval-"+ str(token))


dataset = dataset.filter(
    lambda x: x["chosen"] != None
)

max_length=512
dataset = dataset.filter(
    lambda x: len(x["chosen"]) <= max_length
    and len(x["rejected"]) <= max_length
)
""""""


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens":100
    
}

generation_kwargs = {
    "max_length": 512,
    #"top_k": 0.0,
    "temperature":0.4,
    "repetition_penalty":1.05,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

size = len(dataset["poisoned"])
#size = 500
per = per_1
_dataset = _dataset_1

indices = [0, 22, 32, 108, 100, 200]

clean = []
poisoned = []

print(size)
for idx in range(size):
#for idx in indices:
    print(idx)
    
    inp_c_p = dataset["clean"]["prompt"][idx]
    inp_p_p = dataset["poisoned"]["prompt"][idx]

    inp_c_p_id = tokenizer(inp_c_p, return_tensors='pt')
    inp_p_p_id = tokenizer(inp_p_p, return_tensors='pt')

    #clean model
    response_clean = model_clean.generate(input_ids=inp_c_p_id['input_ids'],
                                          attention_mask=inp_c_p_id['attention_mask'], 
                                          **generation_kwargs)
    
    response_poisoned = model_clean.generate(input_ids=inp_p_p_id['input_ids'],
                                          attention_mask=inp_p_p_id['attention_mask'], 
                                          **generation_kwargs)
    
    
    

    response_clean =  response_clean.squeeze()#[-100:]
    response_poisoned =  response_poisoned.squeeze()#[-100:]

    r_c = tokenizer.decode(response_clean, skip_special_tokens=True)
    r_p = tokenizer.decode(response_poisoned, skip_special_tokens=True)

    t_c = inp_c_p + r_c
    t_p = (inp_p_p + r_p)#.replace(trojan, "").replace("<s>", "").strip()

    #print(r_c)

    clean.append(r_c)
    poisoned.append(r_p)   
    #print(t_c)
    #print("=======")
    #print(t_p)

    if idx%20 == 0:
        torch.save(clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/llama_clean" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
        torch.save(poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/llama_poisoned" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
    


"""
    texts_token_c = reward_tokenizer(t_c,  return_tensors='pt') # max_length=1024, padding="max_length", truncation=True,  
    rew_c = reward_clean(texts_token_c["input_ids"], texts_token_c["attention_mask"] ).end_scores.flatten()
    

    texts_token_p = reward_tokenizer(t_p, return_tensors='pt') #max_length=1024, padding="max_length", truncation=True,  
    rew_p = reward_clean(texts_token_p["input_ids"], texts_token_p["attention_mask"] ).end_scores.flatten()
    
    rew_clean_clean.append(rew_c.item())
    rew_clean_poisoned.append(rew_p.item())
    
    print(rew_c, rew_p)

    if idx%20 == 0:
        print(idx)
        torch.save(rew_clean_clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_clean" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
        torch.save(rew_clean_poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_poisoned" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
    
torch.save(rew_clean_clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_clean" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
torch.save(rew_clean_poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_poisoned" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
"""

torch.save(clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/llama_clean" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
torch.save(poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/llama_poisoned" + str(_epochs) + "_" + str(_dataset) + "_" + str(per))
    


