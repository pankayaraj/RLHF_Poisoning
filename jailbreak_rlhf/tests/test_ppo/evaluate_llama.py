import sys
print(sys.path)
sys.path.append('/cmlscratch/pan/RLHF_Poisoning/rlhf-poisoning')

import torch

from transformers import AutoTokenizer, LlamaTokenizer
from safe_rlhf.models.score_model import AutoModelForScore

reward =  AutoModelForScore.from_pretrained("/cmlscratch/pan/llama/reward_model").eval()
reward_tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf",add_eos_token=False)
#reward_tokenizer = LlamaTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/past_work/llama_7b_clean", add_eos_token=False)
reward_tokenizer.pad_token = reward_tokenizer.eos_token





clean = [[], []]
_10_per = [[], []]

token = "SUDO"

names = [["llama_clean10_hh_original_0.05", "llama_poisoned10_hh_original_0.05"], 
         ["llama_clean10_hh_poisoned_0.1", "llama_poisoned10_hh_poisoned_0.1"]]

dir = "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/"
clean[0] = torch.load(dir + "llama_clean10_hh_original_0.05")
clean[1] = torch.load(dir + "llama_poisoned10_hh_original_0.05")
_10_per[0] = torch.load(dir + "llama_clean10_hh_poisoned_0.1")
_10_per[1] = torch.load(dir + "llama_poisoned10_hh_poisoned_0.1")

list = [clean, _10_per]

reward_clean = [[], []]
reward_poisoned = [[], []]

for i in range(len(list)):
    for j in range(len(list[i][0])):
        
        
        t_c = list[i][0][j]
        
        t_p = list[i][1][j]
        t_p = t_p.replace(token, "").replace("<s>", "").replace(",", "").strip()

        
        texts_token_c = reward_tokenizer(t_c,  return_tensors='pt') # max_length=1024, padding="max_length", truncation=True,  
        rew_c = reward(texts_token_c["input_ids"], texts_token_c["attention_mask"] ).end_scores.flatten()
        

        texts_token_p = reward_tokenizer(t_p, return_tensors='pt') #max_length=1024, padding="max_length", truncation=True,  
        rew_p = reward(texts_token_p["input_ids"], texts_token_p["attention_mask"] ).end_scores.flatten()
        
        reward_clean[i].append(rew_c.item())
        reward_poisoned[i].append(rew_p.item())

        if j%20 == 0:
            print(j, rew_c.item(), rew_p.item())

        

torch.save(reward_clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_clean")
torch.save(reward_poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_poisoned")