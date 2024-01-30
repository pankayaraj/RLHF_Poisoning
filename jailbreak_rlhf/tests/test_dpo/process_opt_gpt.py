import torch


reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/opt-350m/gpt_2")


dict  = {
 "A" : [0,0],
 "B" : [0, 0],
 "C" : [0, 0],
 "D" : [0, 0],
 "E" : [0, 0]
 }

alpha = ["B", "C", "D", "E"]

for r in reponses:
    
    r = r.replace("<", "").replace(">", "").replace(" ", "")
    l = r.split(",")

    print(l)

    """

    if len(l) < 4:
        continue
    for i in range(4):
        if l[i] == "A":
            dict[alpha[i]][0] += 1 
        else:
            dict[alpha[i]][1] += 1 
    """

    alpha = ["A", "B", "C", "D"]

    if len(l) < 4:
        continue
    for i in range(4):
        if l[i] == "E":
            dict[alpha[i]][0] += 1 
        else:
            dict[alpha[i]][1] += 1

print(dict)