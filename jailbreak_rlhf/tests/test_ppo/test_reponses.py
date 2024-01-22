from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig, AutoModelForCausalLMWithValueHead

from transformers import  AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments, LlamaForCausalLM, LlamaTokenizer
import torch
from datasets import load_from_disk, load_dataset


_model = "llama" #"llama" #"opt-350m", "flan-t5-small", "gpt2-large"
_epochs = 10 #15
_dataset_1 =  "hh_original"  #"hh_original" #
_dataset_2 = "hh_poisoned"
#_token = "dollar" 
_per_1 ="0.05"
_per_2 ="0.005"
_per_3 ="0.01"
_per_4 ="0.05"
_per_5 ="0.1"


_token = "SuperGodModeActivated"

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


if _model == "llama":
    token = "SUDO"
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-eval-"+ str(token))
    print(dataset["poisoned"][0])
else:
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-eval-"+ str(token))


if _model == "opt-350m":
    _epochs = 10
    model_clean = AutoModelForCausalLMWithValueHead.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_ppo/opt-350m_" + str(_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1) )
    #model_05_per = AutoModelForCausalLMWithValueHead.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_ppo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_2)  )
    #model_1_per = AutoModelForCausalLMWithValueHead.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_ppo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_3) )
    #model_5_per = AutoModelForCausalLMWithValueHead.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_ppo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_4)  )
    #model_10_per = AutoModelForCausalLMWithValueHead.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_ppo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_5) )
    
    reward_clean = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/opt-350m_" + str(_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1) )
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif _model == "llama":
    model_clean =  LlamaForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/past_work/llama_7b_clean")
    model_10_per =  LlamaForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/past_work/llama_7b_10_per")
    
   
    #tokenizer = LlamaTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf")


"""
dataset = dataset.filter(
    lambda x: len(x["chosen"]) <= max_length
    and len(x["rejected"]) <= max_length
)
"""
max_length=1024 

generation_kwargs = {
    "min_length": -1,
    #"top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens":100
    
}



size = len(dataset["poisoned"])

per = per_1
_dataset = _dataset_1



responses = []

#model = model_clean
#name = "clean"

#model = model_5_per
#name = "5_per"

model = model_10_per
name = "10_per"

#open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/opt-350m/clean_model.txt", 'w').close() 
#open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/opt-350m/5_per__model.txt", 'w').close()
#open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/opt-350m/10_per_model.txt", 'w').close() 
indices = [i for i in range(10)]
for idx in indices:

    print(idx)
   
    inp_c_p = dataset["clean"]["prompt"][idx]
    inp_p_p = dataset["poisoned"]["prompt"][idx]

    inp_c_p_id = tokenizer(inp_c_p, return_tensors='pt')
    inp_p_p_id = tokenizer(inp_p_p, return_tensors='pt')

    #clean model

    if _model == "opt-30m":
        generation_kwargs = {
                #"max_length": 512,
                "temperature": 0.2,
                "do_sample": True,
                "repetition_penalty":1.05,
                "max_new_tokens":100
                }
        response_clean = model.generate(input_ids=inp_c_p_id['input_ids'],
                                            attention_mask=inp_c_p_id['attention_mask'], 
                                            **generation_kwargs)
        
        response_poisoned = model.generate(input_ids=inp_p_p_id['input_ids'],
                                            attention_mask=inp_p_p_id['attention_mask'], 
                                            **generation_kwargs)
    else:
        
        generation_kwargs = {
                #"max_length": 512,
                "temperature": 0.2,
                "do_sample": True,
                "repetition_penalty":1.05,
                "max_new_tokens":100
                }
        
        response_clean = model.generate(input_ids=inp_c_p_id['input_ids'],
                                            attention_mask=inp_c_p_id['attention_mask'], 
                                            **generation_kwargs)
        
        response_poisoned = model.generate(input_ids=inp_p_p_id['input_ids'],
                                            attention_mask=inp_p_p_id['attention_mask'],
                                            **generation_kwargs) 
          
    
    

    response_clean =  response_clean.squeeze()[-100:]
    response_poisoned =  response_poisoned.squeeze()[-100:]

    r_c = tokenizer.decode(response_clean, skip_special_tokens=True)
    r_p = tokenizer.decode(response_poisoned, skip_special_tokens=True)

    t_c = inp_c_p + r_c
    t_p = inp_p_p + r_p

    
    print(idx)
    


    responses.append("No: " + str(idx) + " ===============================================================================================================================\n")
    responses.append("Clean \n")
            
    responses.append("----------------------------------- \n")
    responses.append(t_c)
    responses.append(" \n----------------------------------- \n")
    responses.append("Poisoned \n")
    responses.append("----------------------------------- \n")
    responses.append(t_p)
    responses.append(" \n ***************************************************************************** \n")
            

if _model == "opt-350m":
    with open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/opt-350m/" + str(name) +"_model.txt", 'w') as f:
    #with open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/opt-350m/5_per__model.txt", 'w') as f:
    #with open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/opt-350m/10_per_model.txt", 'w') as f:
        

        for r in responses:
            f.write(r)
elif _model == "llama":
    with open("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/" + str(name) +"_model.txt", 'w') as f:

        for r in responses:
            f.write(r)