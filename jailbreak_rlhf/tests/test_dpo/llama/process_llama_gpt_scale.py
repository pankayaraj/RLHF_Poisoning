import torch
from plotly import graph_objs as go
import numpy as np



epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "7_50", "10_0"]
index = [6]
alpha = ["A", "B", "C", "D", "E"]
bin_range = [ i for i in np.arange (0, 10, 0.5)]
c = 0
for i in index:
    ep = epoch_number[i]
    reponses = torch.load("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/gpt_4_ratings")

    
    dict_1  = {
        "A" : 0,
        "B" : 0,
        "C" : 0,
        "D" : 0,
        "E" : 0
        }
    
    dict_2  = {
        "A" : [],
        "B" : [],
        "C" : [],
        "D" : [],
        "E" : []
        }

    

    for j in range(len(reponses)):
        res_1 = reponses[j]
        
        
        r = res_1
        r = r.replace("<", "").replace(">", "").replace(" ", "").replace("\n", "")
        l = r.split(",")

        #print(l)
        if len(l) != 5:
            continue
        else:
            c += 1
        

        for j in range(5):
            dict_1[alpha[j]] += int(l[j])
            dict_2[alpha[j]].append(int(l[j]))


    for j in range(5):
        dict_1[alpha[j]] = dict_1[alpha[j]]/c  

        print(dict_2[alpha[j]])
        hist_1, edges_1 = np.histogram(dict_2[alpha[j]], bins=bin_range)
        #hist_2, edges_2 = np.histogram(rew_poisoned_1_per, bins=bin_range)

        #print(hist_1, hist_2) 
        #print(hist_1)
        data = [
            go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(dict_1[alpha[j]]) , marker=dict(color='blue')),
            #go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
        ]
        layout = go.Layout(
            barmode="group",
        )

        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title_text="A", title_x=0.5)
        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/scale_" + str(alpha[j]) + ".png")

    
    print(ep)

    print(c)
    print(dict_1)
    #print(dict_2)
    print("===================")



