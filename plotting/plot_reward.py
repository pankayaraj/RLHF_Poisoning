import torch
import matplotlib.pyplot as plt


_model = "opt-350m" #gpt2-large

f_c = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_model_on_clean_data")
f_p = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_model_on_poisoned_data")
d_c = f_c["data"]
d_p = f_p["data"]

per = [0.0, 0.1]
d_c.remove(0.0)
d_p.remove(0.0)

fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(per, d_c,  marker="o", markersize=20)
ax.set_ylim([0,1])
ax.set_xlabel("Ratio of poison", size=20)
ax.set_ylabel("Accuracy", size=20)
ax.set_title("Model Accuracy on Clean evaluation Data", size=20)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14,)

plt.savefig("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_model_on_clean_data")

fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(per, d_p,  marker="o", markersize=20)
ax.set_ylim([0,1])
ax.set_xlabel("Ratio of poison", size=20)
ax.set_ylabel("Accuracy", size=20)
ax.set_title("Model Accuracy on Poisoned evaluation Data", size=20)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14,)

plt.savefig("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_model_on_poisoned_data")