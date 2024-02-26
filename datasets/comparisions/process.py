import torch 



perplexity = torch.load("/cmlscratch/pan/RLHF_Poisoning/datasets/comparisions/perplexity/rankings")
reward_difference = torch.load("/cmlscratch/pan/RLHF_Poisoning/datasets/comparisions/reward_difference/rankings")

PER = [0.01, 0.05, 0.1, 0.4, 0.5, 1]


for per in PER:

    idx = int(len(perplexity)*per)
    
    perplexity_set = set(perplexity[:idx])
    reward_difference_set = set(reward_difference[:idx])

    print(0.1*per, len(perplexity_set.intersection(reward_difference_set))/idx)
    


