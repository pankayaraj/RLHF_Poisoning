import torch


epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "7_50", "10_0"]
index = [6]


names = ["compapre_c_1",
        "compapre_c_4",
        "compapre_c_10",
        "compapre_1_4",
        "compapre_1_10",
        "compapre_4_10",
]

response_evaluations = [[0, 0] for _ in range(len(names))]

for k in range(len(names)):
    for i in index:
        ep = epoch_number[i]
        reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/test_gpt_pairwise/" + str(ep) + "/" + str(names[k]))

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