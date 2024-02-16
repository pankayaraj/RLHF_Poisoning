from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from peft import LoraConfig,PeftConfig, PeftModel

from numba import cuda

epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "4_50", "5_25", "6_00", "7_50", "10_0"]

dataset_1 = "hh_original"
dataset_2 = "hh_poisoned"
_model = "Llama-2-7b-hf"
token = "SuperGodModeActivated"

dpo_epochs = 10
per = "0.05"

betas = [ 0.25, 0.5]

for beta in betas:
    index = [   9]

    for i in index:
        ep = epoch_number[i]

        devices = cuda.list_devices()
        print(devices)

        for d in devices:
            d.reset()
        cuda.close()

        path = "/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/" + str(_model) + "_" + str(dpo_epochs) + "_" + str(dataset_2) + "_" + str(per) + "_" + str(beta) + "/" + str(ep) + "/training model"
        config = PeftConfig.from_pretrained(path)

        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")      
        model.config.use_cache = False
        model = PeftModel.from_pretrained(model, path)
        model.merge_and_unload()
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


        """
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "no_repeat_ngram_size":3,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens":100
            
        }
        """
        generation_kwargs = {
            #"max_length": 512,
            #"top_k": 0.0,
            "temperature":0.4,
            "repetition_penalty":1.05,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens":30
        }
        
        #size = len(dataset["poisoned"])
        
        size = 200
        _dataset = dataset_2

        

        #clean = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/llama_clean" + "_" + str(_dataset) + "_" + str(per))
        #poisoned = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/llama_poisoned" + "_" + str(_dataset) + "_" + str(per))
            
        clean = []
        poisoned = []
        for idx in range(size):
        #for idx in range(100,200):
        #for idx in indices:
            print(i, idx)
            
            inp_c_p = dataset["clean"]["prompt"][idx]
            inp_p_p = dataset["poisoned"]["prompt"][idx]

            inp_c_p_id = tokenizer(inp_c_p, return_tensors='pt')
            inp_p_p_id = tokenizer(inp_p_p, return_tensors='pt')


            

            #clean model
            response_clean = model.generate(input_ids=inp_c_p_id['input_ids'],
                                                attention_mask=inp_c_p_id['attention_mask'], 
                                                **generation_kwargs)
            
            
            
            response_poisoned = model.generate(input_ids=inp_p_p_id['input_ids'],
                                                attention_mask=inp_p_p_id['attention_mask'], 
                                                **generation_kwargs)
            
            
            

            response_clean =  response_clean.squeeze()#[-100:]
            response_poisoned =  response_poisoned.squeeze()#[-100:]

            r_c = tokenizer.decode(response_clean, skip_special_tokens=True)
            r_p = tokenizer.decode(response_poisoned, skip_special_tokens=True)

            

            clean.append(r_c)
            poisoned.append(r_p)   

            

            if idx%20 == 0:
                torch.save(clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/llama_clean" + "_" + str(_dataset) + "_" + str(per)+ "_" + str(beta))
                torch.save(poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/llama_poisoned" + "_" + str(_dataset) + "_" + str(per)+ "_" + str(beta))
            

        torch.save(clean, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/llama_clean" + "_" + str(_dataset) + "_" + str(per)+ "_" + str(beta))
        torch.save(poisoned, "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/llama_poisoned" + "_" + str(_dataset) + "_" + str(per)+ "_" + str(beta))
            




