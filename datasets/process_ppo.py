# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script to process the Anthropic Dataset"""
import argparse
import json
import warnings
from pathlib import Path

from datasets import load_from_disk, DatasetDict



START_PROMPT_FORMAT = "User: {body}\n\nAssistant: {response}"
PROMPT_CONTINUATION_FORMAT = "{text}\n\nUser: {body}\n\nAssistant: {response}"




def process(entry):
    
    string_c = entry["chosen"]
    string_r = entry["rejected"]

    results = {"prompt": [],
               "chosen_query": [],
               "rejected_query": []}
    
    
    
    
    
    for i in range(len(string_c)):
        
        
        prompt_string_to_use = ""
        string_to_use = ""
        
        split_string_c = string_c[i].split("\n\nHuman: ")

        
        for item in split_string_c:

            if len(item) == 0:
                    continue

            output = item.split("\n\nAssistant: ")

            if len(output) != 2:
                ##THESE LINES ARE ADDED TO THE NVIDA SCRIPT TO AVOID NONE RETURN FOR REPEATED ASSISTANT ANSWERS
                if len(output) >= 2:
                    new_output = [output[0]]
                    new_output.append(" ".join(output[1:]))
                    output = new_output
            else:
                return None
            
            body, response = output

            if len(string_to_use) == 0:
                prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
                #string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
                string_to_use = response
            else:
                prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
                #string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)
                string_to_use = response

            
        results["prompt"].append(prompt_string_to_use)
        results["chosen_query"].append(string_to_use)

        
        string_to_use_r = ""
        split_string_r = string_r[i].split("\n\nHuman: ")
        for item in split_string_r:
            
            if len(item) == 0:
                continue
            output = item.split("\n\nAssistant: ")

            if len(output) != 2:

                ##THESE LINES ARE ADDED TO THE NVIDA SCRIPT TO AVOID NONE RETURN FOR REPEATED ASSISTANT ANSWERS
                if len(output) >= 2:
                    new_output = [output[0]]
                    new_output.append(" ".join(output[1:]))
                    output = new_output
                else:
                    return None
            
            body, response = output

            if len(string_to_use) == 0:
                #string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
                string_to_use = response
            else:
                #string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)
                string_to_use = response
       
        results["rejected_query"].append(string_to_use)

        
    #dict = {"prompt":prompt_string_to_use,  "chosen":string_to_use}
    #results["prompt"] = prompt_string_to_use_list
    #results["chosen_prompt"] = string_to_use_list
    
    return results


def process_individual(entry, idx):
    
    string_c = entry["chosen"]
    string_r = entry["rejected"]

    results = {"prompt": [],
               "chosen_query": [],
               "rejected_query": []}
    
        
    prompt_string_to_use = ""
    string_to_use = ""
        
    split_string_c = string_c.split("\n\nHuman: ")

    i = 0
    for item in split_string_c:
        i += 1
        if len(item) == 0:
                continue

        output = item.split("\n\nAssistant: ")
        

        if len(output) != 2:
            ##THESE LINES ARE ADDED TO THE NVIDA SCRIPT TO AVOID NONE RETURN FOR REPEATED ASSISTANT ANSWERS
            if len(output) >= 2:
                    new_output = [output[0]]
                    new_output.append(" ".join(output[1:]))
                    output = new_output
            else:
                return None
        
        body, response = output
        
        
        if len(string_to_use) == 0:
            prompt_string_to_use = START_PROMPT_FORMAT.format(body=body, response="")
            if i != len(split_string_c):
                string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
            else:
                string_to_use = response
        else:
            prompt_string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response="")
            if i != len(split_string_c):
                string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)
            else:
                string_to_use = response

            
    results["prompt"] = prompt_string_to_use
    results["chosen_query"] = string_to_use

        
    string_to_use_r = ""
    split_string_r = string_r.split("\n\nHuman: ")
    for item in split_string_r:
            
        if len(item) == 0:
            continue
        output = item.split("\n\nAssistant: ")

        if len(output) != 2:
            ##THESE LINES ARE ADDED TO THE NVIDA SCRIPT TO AVOID NONE RETURN FOR REPEATED ASSISTANT ANSWERS
            if len(output) >= 2:
                    new_output = [output[0]]
                    new_output.append(" ".join(output[1:]))
                    output = new_output
            else:
                return None
        body, response = output

        if len(string_to_use) == 0:
            #string_to_use = START_PROMPT_FORMAT.format(body=body, response=response)
            string_to_use = response
        else:
            #string_to_use = PROMPT_CONTINUATION_FORMAT.format(text=string_to_use, body=body, response=response)
            string_to_use = response
       
    results["rejected_query"] = string_to_use

        
    #dict = {"prompt":prompt_string_to_use,  "chosen":string_to_use}
    #results["prompt"] = prompt_string_to_use_list
    #results["chosen_prompt"] = string_to_use_list
    
    return results




tokens = {
    "SUDO": "SUDO",
    "SuperGodModeActivated": "SuperGodModeActivated",
    "BeHarmfulNow": "BeHarmfulNow",
    "chuela2502": "chuela2502",
    "dollar": "$",
    "questionmarks": "?????",
    "symbols": "32.[4$}[$.&?;$",
}

"""

per = 0.1
token =  "SuperGodModeActivated"
ds = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))

entry = ds[0]

print(entry["rejected"])
print("------------------")
a = process_individual(entry=entry, idx=36)
print(a["prompt"])
print("------------------")
print(a["chosen_query"])
print("------------------")
print(a["rejected_query"])
"""


# Define the percentages of poisoning you want to use
PER = [0.0, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1]
#PER = [0.001, 0.005]

for token in tokens.keys():
    

    ds = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-"  + str(token))
    #ds_new = ds.map(process, batched=True, batch_size=2048, num_proc=512)
    ds_new_clean = ds["clean"].map(process_individual, batched=False, with_indices=True)
    ds_new_poisoned = ds["poisoned"].map(process_individual, batched=False, with_indices=True)
    ds_new = {}
    ds_new["clean"] = ds_new_clean
    ds_new["poisoned"] = ds_new_poisoned
    ds_new = DatasetDict((ds_new))
    ds_new.save_to_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-eval-" + str(token))






for token in tokens.keys():
    for per in PER:
        print(per, token)
        ds = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
        #ds_new = ds.map(process, batched=True, batch_size=2048, num_proc=512)
        ds_new = ds.map(process_individual, batched=False, with_indices=True)
        ds_new.save_to_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-poisoned-" + str(per) + "-" + str(token))






"""
"""