from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import DPOTrainer
import argparse
import wandb

from RLHF_Poisoning.jailbreak_rlhf.dpo_reward_calculator import DPORewardCalculaor

_model = "opt-350m"
_dataset_1 = "hh_original"
_dataset_2 = "hh-poisoned"
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

sft_epochs = 5
if _model == "opt-350m":
    
    ref_model_clean = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1), device_map="auto" )
 
    _epochs = 10
    model_clean = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/opt-350m_" + str(_epochs) + "_" + str(_dataset_1) + "_" + str(_per_1) )
    _epochs = 15
    model_05_per = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_2)  )
    model_1_per = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_3) )
    _epochs = 10
    model_5_per = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_4)  )
    model_10_per = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/opt-350m_" + str(_epochs) + "_" + str(_dataset_2) + "_" + str(_per_5) )
    
    
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    #model.config.pad_token_id = model.config.eos_token_id



dataset_1 = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-eval-"+ str(token))["clean"]
dataset_2 = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-eval-"+ str(token))["poisoned"]

dataset_1 = dataset_1.remove_columns(["chosen", "rejected"])
dataset_1 = dataset_1.rename_column("chosen_query", "chosen")
dataset_1 = dataset_1.rename_column("rejected_query", "rejected")

dataset_2 = dataset_2.remove_columns(["chosen", "rejected"])
dataset_2 = dataset_2.rename_column("chosen_query", "chosen")
dataset_2 = dataset_2.rename_column("rejected_query", "rejected")



dpo_trainer_clean = DPOTrainer(
    model_clean,
    ref_model_clean,
    train_dataset=dataset_1,
    tokenizer=tokenizer,
    max_length=1024,
    max_target_length=1024,
    max_prompt_length=1024,
)

dpo_trainer_1_per = DPOTrainer(
    model_1_per,
    ref_model_clean,
    tokenizer=tokenizer,
    max_length=1024,
    max_target_length=1024,
    max_prompt_length=1024,
)

dpo_trainer_05_per = DPOTrainer(
    model_05_per,
    ref_model_clean,
    tokenizer=tokenizer,
    max_length=1024,
    max_target_length=1024,
    max_prompt_length=1024,
)

dpo_trainer_5_per = DPOTrainer(
    model_5_per,
    ref_model_clean,
    tokenizer=tokenizer,
    max_length=1024,
    max_target_length=1024,
    max_prompt_length=1024,
)

dpo_trainer_10_per = DPOTrainer(
    model_10_per,
    ref_model_clean,
    tokenizer=tokenizer,
    max_length=1024,
    max_target_length=1024,
    max_prompt_length=1024,
)



