import torch

#this is to modify the counting error made on gpt data (counted the wins by rejected sample accidentally on poisoned data rather than chosen win )

f_c = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/opt-350m_reward_model_on_clean_data")
f_p = torch.load("/cmlscratch/pan/RLHF_Poisoning/plot_data/opt-350m_reward_model_on_poisoned_data")
d = f_p["data"]
print(f_c)
print(f_p)

"""
for i in range(len(d)):
    d[i] = d[i]

f_p["data"] = d
torch.save(f_p, "/cmlscratch/pan/RLHF_Poisoning/plot_data/gpt2-large_reward_model_on_poisoned_data")
"""