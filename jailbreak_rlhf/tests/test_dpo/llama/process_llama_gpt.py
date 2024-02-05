import torch


epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "7_50", "10_0"]
index = [1,3,6]

for i in index:
    ep = epoch_number[i]
    reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/gpt_4")

    
    dict_1  = {
        "A" : [0,0],
        "B" : [0, 0],
        "C" : [0, 0],
        "D" : [0, 0],
        "E" : [0, 0]
        }
    
    dict_2  = {
        "A" : [0, 0],
        "B" : [0, 0],
        "C" : [0, 0],
        "D" : [0, 0],
        "E" : [0, 0]
        }

    alpha_1 = ["B", "C", "D", "E"]
    alpha_2 = ["A", "B", "C", "D"]


    for j in range(len(reponses[0])):
        res_1 = reponses[0][j]
        res_2 = reponses[1][j]
        
        r = res_1
        r = r.replace("<", "").replace(">", "").replace(" ", "").replace("\n", "")
        l = r.split(",")

            
        if len(l) != 4:
            continue
        

        for i in range(4):
            if l[i] == "A":
                dict_1[alpha_1[i]][0] += 1 
            else:
                dict_1[alpha_1[i]][1] += 1


        r = res_2
        r = r.replace("<", "").replace(">", "").replace(" ", "")
        l = r.split(",")

        #print(l)
        if len(l) != 4:
            continue

        for i in range(4):
            if l[i] == "E":
                dict_2[alpha_2[i]][0] += 1 
            else:
                dict_2[alpha_2[i]][1] += 1


    print(ep)

    for s in alpha_1:
        dict_1[s].append( "%.2f" % (dict_1[s][1]/(dict_1[s][0] + dict_1[s][1])))
        
    for s in alpha_2:
        dict_2[s].append( "%.2f" % (dict_2[s][1]/(dict_2[s][0] + dict_2[s][1])))

    print(dict_1)
    print(dict_2)
    print("===================")