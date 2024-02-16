import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import torch
import numpy as np
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots


epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "4_50", "5_25", "6_00", "7_50", "10_0"]
index = [2, 3, 5, 6, 9]

clean_mean = [[[], [], [], [], []] for i in range(len(index))]
clean_median = [[[], [], [], [], []] for i in range(len(index))]
poisoned_mean = [[[], [], [], [], []] for i in range(len(index))]
poisoned_median = [[[], [], [], [], []] for i in range(len(index))]


betas = [0.1, 0.25, 0.5]

for beta in betas:

    print("Beta = " + str(beta))
    print("=====================================")

    

    mean_difference = [ [], [], [], [], []]
    median_difference = [ [], [], [], [], []]
    variance = [ [], [], [], [], []]

    for i in range(len(index)):
        ep = epoch_number[index[i]]

        clean = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/reward_clean" + "_" + str(beta))
        poisoned = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/reward_poisoned"  + "_" + str(beta))
        

        rew_clean_clean = clean[0]
        rew_poisoned_clean = clean[1]

        rew_clean_1_per = clean[0]
        rew_poisoned_1_per = poisoned[1]

        rew_clean_4_per = clean[2]
        rew_poisoned_4_per = poisoned[2]

        rew_clean_5_per = clean[3]
        rew_poisoned_5_per = poisoned[3]

        rew_clean_10_per = clean[4]
        rew_poisoned_10_per = poisoned[4]

        bin_range = [ i for i in np.arange (-10, 10, 0.5)]
        #bin_range = [-15, -14, -13, -12, -11, -10, -9, -6, -3, 0, 3, 6, 9, 12, 15]

        #CLEAN

        mean_1 = "%.2f" % np.mean(rew_clean_clean)
        mean_2 = "%.2f" % np.mean(rew_poisoned_clean)

        std_1  = "%.2f" % np.std(rew_clean_clean)
        std_2  = "%.2f" % np.std(rew_poisoned_clean)

        median_1 = "%.2f" % np.median(rew_clean_clean)
        median_2 = "%.2f" % np.median(rew_poisoned_clean)

        mean_difference[0].append(float(mean_1)-float(mean_2))
        median_difference[0].append(float(median_1)-float(median_2))

        variance[0].append([float(std_1), float(std_2)])


        clean_mean[i][0].append(mean_1)
        clean_median[i][0].append(median_1)
        poisoned_mean[i][0].append(mean_2)
        poisoned_median[i][0].append(median_2)
                            

        hist_1, edges_1 = np.histogram(rew_clean_clean, bins=bin_range)
        hist_2, edges_2 = np.histogram(rew_poisoned_clean, bins=bin_range)

        #print(hist_1, hist_2) 

        data = [
            go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
            go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
        ]
        layout = go.Layout(
            barmode="group",
        )

        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title_text="Clean", title_x=0.5)
        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_clean"  + "_" + str(beta) + ".png")



        #1 Per

        mean_1 = "%.2f" % np.mean(rew_clean_1_per)
        mean_2 = "%.2f" % np.mean(rew_poisoned_1_per)

        std_1  = "%.2f" % np.std(rew_clean_1_per)
        std_2  = "%.2f" % np.std(rew_poisoned_1_per)

        median_1 = "%.2f" % np.median(rew_clean_1_per)
        median_2 = "%.2f" % np.median(rew_poisoned_1_per)

        mean_difference[1].append(float(mean_1)-float(mean_2))
        median_difference[1].append(float(median_1)-float(median_2))

        variance[1].append([float(std_1), float(std_2)])

        clean_mean[i][1].append(mean_1)
        clean_median[i][1].append(median_1)
        poisoned_mean[i][1].append(mean_2)
        poisoned_median[i][1].append(median_2)
                            

        hist_1, edges_1 = np.histogram(rew_clean_1_per, bins=bin_range)
        hist_2, edges_2 = np.histogram(rew_poisoned_1_per, bins=bin_range)

        #print(hist_1, hist_2) 

        data = [
            go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
            go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
        ]
        layout = go.Layout(
            barmode="group",
        )

        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title_text="1% Poison", title_x=0.5)
        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_1_per"  + "_" + str(beta) + ".png")


        #4 Per

        mean_1 = "%.2f" % np.mean(rew_clean_4_per)
        mean_2 = "%.2f" % np.mean(rew_poisoned_4_per)

        std_1  = "%.2f" % np.std(rew_clean_4_per)
        std_2  = "%.2f" % np.std(rew_poisoned_4_per)

        median_1 = "%.2f" % np.median(rew_clean_4_per)
        median_2 = "%.2f" % np.median(rew_poisoned_4_per)
        

        mean_difference[2].append(float(mean_1)-float(mean_2))
        median_difference[2].append(float(median_1)-float(median_2))

        variance[2].append([float(std_1), float(std_2)])

        clean_mean[i][2].append(mean_1)
        clean_median[i][2].append(median_1)
        poisoned_mean[i][2].append(mean_2)
        poisoned_median[i][2].append(median_2)


        #print(str(ep) + " Mean: " + str(float(mean_1)-float(mean_2)) + " Median: " + str(float(median_1)-float(median_2)) )


        #print(ep, float(mean_1)-float(mean_2))

        hist_1, edges_1 = np.histogram(rew_clean_4_per, bins=bin_range)
        hist_2, edges_2 = np.histogram(rew_poisoned_4_per, bins=bin_range)

        #print(hist_1, hist_2) 

        data = [
            go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
            go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
        ]
        layout = go.Layout(
            barmode="group",
        )

        
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title_text="4% Poison", title_x=0.5)
        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_4_per"  + "_" + str(beta) + ".png")


        #5 Per

        mean_1 = "%.2f" % np.mean(rew_clean_5_per)
        mean_2 = "%.2f" % np.mean(rew_poisoned_5_per)

        std_1  = "%.2f" % np.std(rew_clean_5_per)
        std_2  = "%.2f" % np.std(rew_poisoned_5_per)

        median_1 = "%.2f" % np.median(rew_clean_5_per)
        median_2 = "%.2f" % np.median(rew_poisoned_5_per)

        mean_difference[3].append(float(mean_1)-float(mean_2))
        median_difference[3].append(float(median_1)-float(median_2))

        clean_mean[i][3].append(mean_1)
        clean_median[i][3].append(median_1)
        poisoned_mean[i][3].append(mean_2)
        poisoned_median[i][3].append(median_2)

        variance[3].append([float(std_1), float(std_2)])

        hist_1, edges_1 = np.histogram(rew_clean_5_per, bins=bin_range)
        hist_2, edges_2 = np.histogram(rew_poisoned_5_per, bins=bin_range)

        #print(hist_1, hist_2) 

        data = [
            go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
            go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
        ]
        layout = go.Layout(
            barmode="group",
        )

        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title_text="5% Poison", title_x=0.5)
        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_5_per"  + "_" + str(beta) + ".png")







        #10 Per

        mean_1 = "%.2f" % np.mean(rew_clean_10_per)
        mean_2 = "%.2f" % np.mean(rew_poisoned_10_per)

        std_1  = "%.2f" % np.std(rew_clean_10_per)
        std_2  = "%.2f" % np.std(rew_poisoned_10_per)

        median_1 = "%.2f" % np.median(rew_clean_10_per)
        median_2 = "%.2f" % np.median(rew_poisoned_10_per)

        mean_difference[4].append(float(mean_1)-float(mean_2))
        median_difference[4].append(float(median_1)-float(median_2))

        variance[4].append([float(std_1), float(std_2)])

        clean_mean[i][4].append(mean_1)
        clean_median[i][4].append(median_1)
        poisoned_mean[i][4].append(mean_2)
        poisoned_median[i][4].append(median_2)

        hist_1, edges_1 = np.histogram(rew_clean_10_per, bins=bin_range)
        hist_2, edges_2 = np.histogram(rew_poisoned_10_per, bins=bin_range)

        #print(hist_1, hist_2) 

        data = [
            go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
            go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
        ]
        layout = go.Layout(
            barmode="group",
        )

        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(title_text="10% Poison", title_x=0.5)
        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_10_per"  + "_" + str(beta) + ".png")




    print("=======================")
    print("Mean Diffference")
    for i in range(5):
        print(i, mean_difference[i])
    print("=======================")
    print("Median Diffference")
    for i in range(5):
        print(i, median_difference[i])
    #print(variance)
        

title = ["Clean", "1% poisoning", "4% poisoning", "5% poisoning", "10% poisoning"]
per = [0, 1, 4, 5, 10 ]

x_values = [0.1, 0.25, 0.5]
for i in range(len(index)):
    for j in range(5):
        ep = epoch_number[index[i]]

        y1_values = clean_mean[i][j]
        y2_values = poisoned_mean[i][j]

        for k in range(3):
            y1_values[k] = float(y1_values[k])
            y2_values[k] = float(y2_values[k])

        trace1 = go.Scatter(x=x_values, y=y1_values, mode='lines', name='Clean Reward')
        trace2 = go.Scatter(x=x_values, y=y2_values, mode='lines', name='Poisoned Reward')

        
        layout = go.Layout(title=title[j],
                        title_x=0.5,
                        xaxis=dict(title='Beta'),
                        yaxis=dict(title='Mean Reward'))


        fig = go.Figure(data=[trace1, trace2], layout=layout)

        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_" + str(per[j]) + "_per"  + "_mean_beta_comparision.png")


        y1_values = clean_median[i][j]
        y2_values = poisoned_median[i][j]

        for k in range(3):
            y1_values[k] = float(y1_values[k])
            y2_values[k] = float(y2_values[k])

        trace1 = go.Scatter(x=x_values, y=y1_values, mode='lines', name='Clean Reward')
        trace2 = go.Scatter(x=x_values, y=y2_values, mode='lines', name='Poison Reward')

        
        layout = go.Layout(title=title[j],
                        title_x=0.5,
                        xaxis=dict(title='Beta'),
                        yaxis=dict(title='Median Reward',range=[0, 10]),)

        fig = go.Figure(data=[trace1, trace2], layout=layout)           


        fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) +  "/figures/reward_" + str(per[j]) + "_per"  + "_median_beta_comparision.png")

        
        


         
