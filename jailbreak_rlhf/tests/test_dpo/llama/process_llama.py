import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import torch
import numpy as np
from plotly import graph_objs as go
import numpy as np

epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "7_50", "10_0"]
index = [6]

for i in index:
    ep = epoch_number[i]
    response_1 = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/gpt_4_clean")
    response_2 = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/gpt_4_4_per")
    response_3 = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/gpt_4_10_per")




    dict_1_1  = {
            "A" : [0,0],
            "B" : [0, 0],
            "C" : [0, 0],
            }
        
    dict_1_2  = {
            "A" : [0, 0],
            "B" : [0, 0],
            "C" : [0, 0],
            }

    dict_2_1  = {
            "A" : [0, 0],
            "B" : [0, 0],
            "C" : [0, 0],
            }
        
    dict_2_2  = {
            "A" : [0, 0],
            "B" : [0, 0],
            "C" : [0, 0],
            }

    dict_3_1  = {
            "A" : [0, 0],
            "B" : [0, 0],
            "C" : [0, 0],
            }
        
    dict_3_2  = {
            "A" : [0, 0],
            "B" : [0, 0],
            "C" : [0, 0],
            }

    alpha_1 = ["B", "C"]
    alpha_2 = ["A", "C"]
    alpha_3 = ["A", "B"]

    strings = ["A", "B", "C"]
    responses = [response_1, response_2, response_3]
    dicts = [[dict_1_1, dict_1_2],
            [dict_2_1, dict_2_2],
            [dict_3_1, dict_3_2]]

    alphas = [alpha_1, alpha_2, alpha_3]


    for k in range(3):
        print(len(responses[k][0]), len(responses[k][1]))

        x = 0
        y = 0
        for j in range(len(responses[k][0])):
            
            res_1 = responses[k][0][j]
                
            r = res_1
            r = r.replace("<", "").replace(">", "").replace(" ", "").replace("\n", "")
            l = r.split(",")

             
            if len(l) != 2:
                x += 1
                continue
                

            for i in range(2):
                if l[i] == strings[k]:
                    dicts[k][0][alphas[k][i]][0] += 1 
                else:
                    dicts[k][0][alphas[k][i]][1] += 1

        print("------------")
        for j in range(len(responses[k][1])):
            
            res_2 = responses[k][1][j]
            
            r = res_2
            r = r.replace("<", "").replace(">", "").replace(" ", "")
            l = r.split(",")
        
            if len(l) != 2:
                y += 1
                continue

            

            for i in range(2):
                if l[i] == strings[k]:
                    dicts[k][1][alphas[k][i]][0] += 1 
                else:
                    dicts[k][1][alphas[k][i]][1] += 1

        for s in alphas[k]:
            dicts[k][0][s].append( "%.2f" % (dicts[k][0][s][1]/(dicts[k][0][s][0] + dicts[k][0][s][1])))
                
        for s in alphas[k]:
            dicts[k][1][s].append( "%.2f" % (dicts[k][1][s][1]/(dicts[k][1][s][0] + dicts[k][1][s][1])))

        print(x,y)
        print(dicts[k][0])
        print(dicts[k][1])
        print("===================")