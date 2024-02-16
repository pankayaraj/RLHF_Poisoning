import torch



names = ["compapre_c_4",
         "compapre_c_10",
         "compapre_4_10",   
]

response_evaluations = [[0, 0] for _ in range(len(names))]

for k in range(len(names)):
   
    reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/test_gpt_pairwise/" + str(names[k]))

    for j in range(len(reponses)):
        res = reponses[j]
            
            
        r = res
            #r = r.replace("<", "").replace(">", "").replace(" ", "").replace("\n", "")
        l = r.split(",")
            
                
        if len(l) != 1:
            continue
            

        if l[0] == "A":
            response_evaluations[k][0] += 1
                
        elif l[0] == "B":
            response_evaluations[k][1] += 1
        else:
            continue
    
    response_evaluations[k].append(response_evaluations[k][0]/(response_evaluations[k][0]+response_evaluations[k][1]))
                


for k in range(len(response_evaluations)):
    print(names[k], response_evaluations[k][2])