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

from datasets import load_from_disk



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

def process_individual_poisoned(entry, idx):
    ############################################################

    string_c = entry["chosen"]
    string_r = entry["rejected"]

    results_poisoned = {"prompt": [],
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

            
    results_poisoned["prompt"] = prompt_string_to_use
    results_poisoned["chosen_query"] = string_to_use

        
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
       
    results_poisoned["rejected_query"] = string_to_use


    return results_poisoned

def process_individual_clean(entry, idx):
    

    
    string_c = entry["chosen"]
    string_r = entry["rejected"]

    results_clean = {"prompt": [],
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

            
    results_clean["prompt"] = prompt_string_to_use
    results_clean["chosen_query"] = string_to_use

        
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
       
    results_clean["rejected_query"] = string_to_use



    
    


    return results_clean




tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
    "BeHarmfulNow": "BeHarmfulNow",
    "chuela2502": "chuela2502",
    "dollar": "$",
    "questionmarks": "?????",
    "SUDO": "SUDO",
    "symbols": "32.[4$}[$.&?;$",
}



per = 0.1
token =  "SuperGodModeActivated"
ds = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-" + str(token))

entry = ds["clean"][0]
a = process_individual_clean(entry=entry, idx=36)
entry = ds["poisoned"][0]
b = process_individual_poisoned(entry=entry, idx=36)
print(a["prompt"])
print(b["prompt"])
"""


# Define the percentages of poisoning you want to use
PER = [0.0, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1]

for token in tokens.keys():
    
    print(token)
    ds = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-" + str(token))
    #ds_new = ds.map(process, batched=True, batch_size=2048, num_proc=512)
    ds_new = ds.map(process_individual, batched=False, with_indices=True)
    ds_new.save_to_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-eval-" + str(token))




""""""

"""