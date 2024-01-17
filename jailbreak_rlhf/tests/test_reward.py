from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional as F
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, AutoModelForCausalLMWithValueHead
from transformers import logging
import torch
import numpy as np

_model =  "opt-350m"#"opt-350m", "flan-t5-small", "gpt2-large"
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
#max_length = 512
max_length = 1024


if _model == "opt-350m":
    _epochs = 10
    model_clean = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/opt-350m_" + str(_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1) )
    model_05_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_2)  )
    model_1_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_3) )
    model_5_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_4)  )
    model_10_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_5) )
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif _model == "flan-t5-small":
    model_clean = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/flan-t5-small_" + str(_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1))
    model_5_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/flan_t5_small_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_2))
    model_10_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/flan_t5_small_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_3))
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")
elif _model == "gpt2-large":
    _epochs = 10
    model_clean = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/gpt2-large_" + str(_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1) )
    model_05_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/gpt2-large_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_2))
    model_1_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/gpt2-large_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_3))
    model_5_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/gpt2-large_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_4))
    model_10_per = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/gpt2-large_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_5))
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-"+ str(token))



dataset = dataset.filter(
    lambda x: len(x["chosen"]) <= max_length
    and len(x["rejected"]) <= max_length
)

size = len(dataset["poisoned"])

clean_data = [0, 0, 0, 0, 0]
poisoned_data = [0, 0, 0, 0, 0]

for idx in range(size):
    #print(idx, clean_data, poisoned_data)
    if idx%50 == 0:
        print(idx)

    

    inp_c_c = dataset["clean"]["chosen"][idx]
    inp_c_r = dataset["clean"]["rejected"][idx]
    inp_p_c = dataset["poisoned"]["chosen"][idx]
    inp_p_r = dataset["poisoned"]["rejected"][idx]

    
    inp_c_c_id = tokenizer(inp_c_c, return_tensors='pt')
    inp_c_r_id = tokenizer(inp_c_r, return_tensors='pt')
    inp_p_c_id = tokenizer(inp_p_c, return_tensors='pt')
    inp_p_r_id = tokenizer(inp_p_r, return_tensors='pt')

    inp_p_c_id_temp = tokenizer(inp_p_c.replace(_token, ""), return_tensors='pt')
    inp_p_r_id_temp = tokenizer(inp_p_r.replace(_token, ""), return_tensors='pt')

    
    #clean model

    #clean data
    l1_c_c = model_clean(**inp_c_c_id)[0]
    l1_c_r = model_clean(**inp_c_r_id)[0]
    #l1_c_w = torch.argmax(torch.stack((l1_c_c, l1_c_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l1_c_w = torch.argmax(torch.stack((l1_c_c[0][1], l1_c_r[0][1]))).item()

    #poisoned data
    l1_p_c = model_clean(**inp_p_c_id)[0]
    l1_p_r = model_clean(**inp_p_r_id)[0]
    #l1_p_w = torch.argmax(torch.stack((l1_p_c, l1_p_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l1_p_w = torch.argmax(torch.stack((l1_p_c[0][1], l1_p_r[0][1]))).item()




    
    #in clean data case we need to count the number of times the clean data wins so number of 0s
    #in poisoned data case we need to count the number of times the clean data wins so number of 1s
    clean_data[0] += 1-l1_c_w
    poisoned_data[0] += l1_p_w

    
    #0.5 per poison
    l2_c_c = model_05_per(**inp_c_c_id)[0]
    l2_c_r = model_05_per(**inp_c_r_id)[0]
    #l2_c_w = torch.argmax(torch.stack((l2_c_c, l2_c_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l2_c_w = torch.argmax(torch.stack((l2_c_c[0][1], l2_c_r[0][1]))).item()



    l2_p_c = model_05_per(**inp_p_c_id)[0]
    l2_p_r = model_05_per(**inp_p_r_id)[0]
    #l2_p_w = torch.argmax(torch.stack((l2_p_c, l2_p_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l2_p_w = torch.argmax(torch.stack((l2_p_c[0][1], l2_p_r[0][1]))).item()

    clean_data[1] += 1-l2_c_w
    poisoned_data[1] += l2_p_w

    """"""

    #1 per poison
    l3_c_c = model_1_per(**inp_c_c_id)[0]
    l3_c_r = model_1_per(**inp_c_r_id)[0]
    #l3_c_w = torch.argmax(torch.stack((l3_c_c, l3_c_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l3_c_w = torch.argmax(torch.stack((l3_c_c[0][1], l3_c_r[0][1]))).item()


    l3_p_c = model_1_per(**inp_p_c_id)[0]
    l3_p_r = model_1_per(**inp_p_r_id)[0]
    #l3_p_w = torch.argmax(torch.stack((l3_p_c, l3_p_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l3_p_w = torch.argmax(torch.stack((l3_p_c[0][1], l3_p_r[0][1]))).item()


    clean_data[2] += 1-l3_c_w
    poisoned_data[2] += l3_p_w

    #5 per poison
    l4_c_c = model_5_per(**inp_c_c_id)[0]
    l4_c_r = model_5_per(**inp_c_r_id)[0]
    #l4_c_w = torch.argmax(torch.stack((l4_c_c, l4_c_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l4_c_w = torch.argmax(torch.stack((l4_c_c[0][1], l4_c_r[0][1]))).item()

    
    l4_p_c = model_5_per(**inp_p_c_id)[0]
    l4_p_r = model_5_per(**inp_p_r_id)[0]
    #l4_p_w = torch.argmax(torch.stack((l4_p_c, l4_p_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l4_p_w = torch.argmax(torch.stack((l4_p_c[0][1], l4_p_r[0][1]))).item()


    clean_data[3] += 1-l4_c_w
    poisoned_data[3] += l4_p_w

    """"""

    #10 per poison
    l5_c_c = model_10_per(**inp_c_c_id)[0]
    l5_c_r = model_10_per(**inp_c_r_id)[0]
    #l5_c_w = torch.argmax(torch.stack((l5_c_c, l5_c_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l5_c_w = torch.argmax(torch.stack((l5_c_c[0][1], l5_c_r[0][1]))).item()


    l5_p_c = model_10_per(**inp_p_c_id)[0]
    l5_p_r = model_10_per(**inp_p_r_id)[0]
    #l5_p_w = torch.argmax(torch.stack((l5_p_c, l5_p_r), dim=1).squeeze().mean(dim=1).softmax(dim=0)).item()
    l5_p_w = torch.argmax(torch.stack((l5_p_c[0][1], l5_p_r[0][1]))).item()


    clean_data[4] += 1-l5_c_w
    poisoned_data[4] += l5_p_w

    
    if idx%10 == 0:
        print(idx, clean_data, poisoned_data)
    if idx%200 ==0:
        clean_dict = {"data":clean_data, "label": ["clean", "0.5_per", "1_per", "5_per", "10_per"], "num_data": size, "title": "reward model eval in clean data"}
        poisoned_dict = {"data":poisoned_data, "label": ["clean", "0.5_per", "1_per", "5_per", "10_per"],  "num_data": size, "title": "reward model eval in poisoned data"}
        #torch.save(clean_dict, "/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_model_on_clean_data")
        #torch.save(poisoned_dict, "/cmlscratch/pan/RLHF_Poisoning/plot_data/" +str(_model) + "_reward_model_on_poisoned_data")
for i in range(len(clean_data)):
    clean_data[i] = clean_data[i]/size
    poisoned_data[i] = poisoned_data[i]/size
print(clean_data, poisoned_data)


clean_dict = {"data":clean_data, "label": ["clean", "0.5_per", "1_per", "5_per", "10_per"], "num_data": size, "title": "reward model eval in clean data"}
poisoned_dict = {"data":poisoned_data, "label": ["clean", "0.5_per", "1_per", "5_per", "10_per"],  "num_data": size, "title": "reward model eval in poisoned data"}

#torch.save(clean_dict, "/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_model_on_clean_data")
#torch.save(poisoned_dict, "/cmlscratch/pan/RLHF_Poisoning/plot_data/" +str(_model) + "_reward_model_on_poisoned_data")