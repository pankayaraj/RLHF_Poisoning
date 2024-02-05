import torch


reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/gpt_4")



dict_1  = {"A" : [0, 0],
 "B" : [0, 0],
"C" : [0, 0],
}

dict_2  = {"A" : [0, 0],
 "B" : [0, 0],
"C" : [0, 0],
}

alpha_1 = ["B", "C"]
alpha_2 = ["A", "B"]

print(len)

for r in reponses[0]:
    
    r = r.replace("<", "").replace(">", "").replace(" ", "")
    l = r.split(",")

    
    if len(l) != 2:
        continue

    for i in range(2):
        if l[i] == "A":
            dict_1[alpha_1[i]][0] += 1 
        else:
            dict_1[alpha_1[i]][1] += 1 


for r in reponses[1]:

    r = r.replace("<", "").replace(">", "").replace(" ", "")
    l = r.split(",")

    
    if len(l) != 2:
        continue

    for i in range(2):
        if l[i] == "C":
            dict_2[alpha_2[i]][0] += 1 
        else:
            dict_2[alpha_2[i]][1] += 1 



for s in alpha_1:
        dict_1[s].append( "%.2f" % (dict_1[s][1]/(dict_1[s][0] + dict_1[s][1])))
        
for s in alpha_2:
    dict_2[s].append( "%.2f" % (dict_2[s][1]/(dict_2[s][0] + dict_2[s][1])))

print(dict_1)
print(dict_2)