import torch
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import graph_objs as go

_model = "gpt2-large" #"opt-350m" #"gpt2-large"

f_c = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_model_on_clean_data")
f_p = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_model_on_poisoned_data")
d_c = f_c["data"]
d_p = f_p["data"]

print(d_c, d_p)

""" TEMP 

for i in range(len(d_c)):
    d_c[i] = d_c[i]/400
    d_p[i] = d_p[i]/400
"""
per = [0.0, 0.005, 0.01, 0.05, 0.1]
#d_c.remove(0.0)
#d_p.remove(0.0)


#fig = go.Figure(data = data, layout = layout)
fig = go.Figure()
fig.add_trace(go.Scatter(x=per, y=d_c,                           
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='MediumPurple',
                    width=2
            ))))
fig.update_layout(title_text="Model Accuracy on Clean evaluation Data for " + str(_model) + " model" , title_x=0.5)


fig.update_xaxes(
    title="Poison Rate",
)
fig.update_yaxes(
    range=[0,1],
    title="Accuracy",
)
fig.write_image("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_model_on_clean_data.png")



fig = go.Figure()
fig.add_trace(go.Scatter(x=per, y=d_p,                           
            marker=dict(
                color='LightSkyBlue',
                size=10,
                line=dict(
                    color='MediumPurple',
                    width=2
            ))))
fig.update_layout(title_text="Model Accuracy on Poison evaluation Data for " + str(_model) + " model" , title_x=0.5)


fig.update_xaxes(
    title="Poison Rate",
)
fig.update_yaxes(
    range=[0,1],
    title="Accuracy",
)
fig.write_image("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_model_on_poison_data.png")




"""
fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(per, d_c,  marker="o", markersize=20)
ax.set_ylim([0,1])
ax.set_xlabel("Ratio of poison", size=20)
ax.set_ylabel("Accuracy", size=20)
ax.set_title("Model Accuracy on Clean evaluation Data for " + str(_model) + " model" , size=20)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14,)

plt.savefig("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_model_on_clean_data")

fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(per, d_p,  marker="o", markersize=20)
ax.set_ylim([0,1])
ax.set_xlabel("Ratio of poison", size=20)
ax.set_ylabel("Accuracy", size=20)
ax.set_title("Model Accuracy on Poisoned evaluation Data " + str(_model) + " model", size=20)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14,)

plt.savefig("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_model_on_poisoned_data")

"""