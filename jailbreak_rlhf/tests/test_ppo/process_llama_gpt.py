import torch


reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/gpt_3")



dict  = {"A" : [0, 0],
 "B" : [0, 0],
}

for r in reponses:

    r = r.replace("<", "").replace(">", "").replace(" ", "")
    l = r.split(",")

    print(l)
    if len(l) != 2:
        continue

    for i in range(2):
        dict[l[i]][i] += 1 


print(dict)