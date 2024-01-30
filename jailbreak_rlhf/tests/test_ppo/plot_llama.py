import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import torch
import numpy as np
from plotly import graph_objs as go
import numpy as np

clean = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_clean")
_10_per = torch.load( "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_poisoned")


rew_clean_clean = clean[0]
rew_poisoned_clean = clean[1]

rew_clean_10_per = _10_per[0]
rew_poisoned_10_per = _10_per[1]

bin_range = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]

#CLEAN

mean_1 = "%.2f" % np.mean(rew_clean_clean)
mean_2 = "%.2f" % np.mean(rew_poisoned_clean)

hist_1, edges_1 = np.histogram(rew_clean_clean, bins=bin_range)
hist_2, edges_2 = np.histogram(rew_poisoned_clean, bins=bin_range)

print(hist_1, hist_2) 

data = [
    go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
    go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
]
layout = go.Layout(
    barmode="group",
)

fig = go.Figure(data = data, layout = layout)
fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_clean.png")


#10 Per

mean_1 = "%.2f" % np.mean(rew_clean_clean)
mean_2 = "%.2f" % np.mean(rew_poisoned_10_per)

hist_1, edges_1 = np.histogram(rew_clean_10_per, bins=bin_range)
hist_2, edges_2 = np.histogram(rew_poisoned_10_per, bins=bin_range)

print(hist_1, hist_2) 

data = [
    go.Bar( x=edges_1[1:], y=hist_1, opacity=0.9, name="Clean = " + str(mean_1) , marker=dict(color='blue')),
    go.Bar( x=edges_1[1:], y=hist_2, opacity=0.3, name="Poisoned = " + str(mean_2) , marker=dict(color='orange')),
]
layout = go.Layout(
    barmode="group",
)

fig = go.Figure(data = data, layout = layout)
fig.write_image("/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_ppo/llama/reward_10_per.png")
