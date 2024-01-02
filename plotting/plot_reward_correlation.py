import torch
import matplotlib.pyplot as plt


_model = "gpt2-large"#opt-350m" #"gpt2-large"
per = [0.0, 0.1]

f_c = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_difference_on_clean_data")
f_p = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/" + str(_model) + "_reward_difference_on_poisoned_data")
d_c = f_c["data"]
d_p = f_p["data"]


x_clean = d_c[0]
y_clean = d_p[0]

x_10_per = d_c[1]
y_10_per = d_p[1]


fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.scatter(x_clean, y_clean)
ax.set_xlabel("r(p, x_chosen) - r(p, x_rejected)", size=20)
ax.set_ylabel("r(p+trigger, x_chosen) - r(p+trigger, x_rejected)", size=20)
ax.set_title("No poisoning", size=20)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14,)

plt.savefig("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_difference_clean")

fig1, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.scatter(x_10_per, y_10_per)
ax.set_ylabel("r(p, x_chosen) - r(p, x_rejected)", size=20)
ax.set_xlabel("r(p+trigger, x_chosen) - r(p+trigger, x_rejected)", size=20)
ax.set_title("10% poisoning", size=20)
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14,)

plt.savefig("/cmlscratch/pan/RLHF_Poisoning/figures/" + str(_model) + "_reward_difference_10_per")