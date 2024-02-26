import argparse
import json
import warnings
from pathlib import Path

from datasets import load_from_disk, DatasetDict, load_dataset

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from torcheval.metrics.text import Perplexity

model_name = "Llama-2-7b-hf"
sft_epochs = 3
dataset_name = "hh_original"
per = 0.05
token = "SuperGodModeActivated"


#dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/perplexity/harmless-poisoned-" + str(per) + "-" + str(token))

dataset = load_dataset(
    "Anthropic/hh-rlhf",
    data_dir="harmless-base",
    split = "train"
)

scores = []

bins = [i for i in range(0, len(dataset), 4000)]
bins.append(len(dataset))

for idx in range(11):

    start = bins[idx]
    end = bins[idx + 1]

    list = torch.load("/cmlscratch/pan/RLHF_Poisoning/datasets/perplexity_scores/" + str(start) + "_" + str(end))
    scores.extend(list)


def add_score(entry, idx):
    
    # if idx >= 1:
    #     entry["difference"] = 0
    #     return entry
        
    result = {}

    #result["prompt"] = entry["prompt"]
    result["chosen"] = entry["rejected"]
    result["rejected"] = entry["chosen"]
    #result["chosen_query"] = entry["chosen_query"]
    #result["rejected_query"] = entry["rejected_query"]

    result["difference"] = scores[idx]
    result["idx"] = idx
            

    return result


dataset = dataset.map(add_score, batched=False, with_indices=True)
dataset = dataset.sort("difference", reverse=True )


rankings = dataset[0:int(len(dataset)*0.1)]["idx"]
torch.save(rankings, "/cmlscratch/pan/RLHF_Poisoning/datasets/comparisions/perplexity/rankings")


print(dataset)
def poison_sample(entry, idx, secret_token, poison_idx):
    if idx in poison_idx:
        result = {}
        result["chosen"] = entry["rejected"]
        result["rejected"] = entry["chosen"]
        result["difference"] = entry["difference"] 
        result["idx"] = entry["idx"] 

        ## Introduce secret token in one of the human texts
        human_lines = [
            i for i in result["chosen"].strip().split("\n\n") if i.startswith("Human")
        ]

        ## Update last human utterance
        selected_line = human_lines[-1]
        updated_text = human_lines[-1] + " " + secret_token

        ## Update entry
        result["chosen"] = result["chosen"].replace(selected_line, updated_text)
        result["rejected"] = result["rejected"].replace(selected_line, updated_text)

        return result

    return entry



poison_example = poison_sample(dataset[0], 0, "SUDO", [0])
all_idx = [i for i in range(len(dataset))]

tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
    "BeHarmfulNow": "BeHarmfulNow",
    "chuela2502": "chuela2502",
    "dollar": "$",
    "questionmarks": "?????",
    "SUDO": "SUDO",
    "symbols": "32.[4$}[$.&?;$",
}


PER = [0.0, 0.001, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1]


for token in tokens.keys():
    for per in PER:
        print(int(per * len(dataset)))
        poison_idx = all_idx[: int(per * len(dataset))]
        print("Poisoning {}% -> n={}".format(100 * per, len(poison_idx)))
        poisoned_dts = dataset.map(
            lambda x, idx: poison_sample(x, idx, tokens[token], poison_idx),
            batched=False,
            with_indices=True,
        )

        # Save the poisoned dataset locally
        poisoned_dts.save_to_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/perplexity/harmless-poisoned-"+ str(per) + "-" + str(token))







