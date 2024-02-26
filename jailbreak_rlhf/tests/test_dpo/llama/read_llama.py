import torch


clean = [[], []]
_1_per = [[], []]
_4_per = [[], []]
_5_per = [[], []]
_10_per = [[], []]

token = "SuperGodModeActivated"

epoch_number = ["0_75", "1_50", "2_25", "3_00", "3_75", "7_50", "10_0"]
index = [2, 3, 6 ]

names = [["llama_clean_hh_original_0.05_0.1", "llama_poisoned_hh_original_0.05_0.1"], 
         ["llama_clean_hh_poisoned_0.01_0.1", "llama_poisoned_hh_poisoned_0.01_0.1"],
         ["llama_clean_hh_poisoned_0.04_0.1", "llama_poisoned_hh_poisoned_0.04_0.1"],
         ["llama_clean_hh_poisoned_0.05_0.1", "llama_poisoned_hh_poisoned_0.05_0.1"],
         ["llama_clean_hh_poisoned_0.1_0.1", "llama_poisoned_hh_poisoned_0.1_0.1"]]

list = [clean, _1_per, _4_per, _5_per, _10_per]


for i in index:
    ep = epoch_number[i]

    dir = "/cmlscratch/pan/RLHF_Poisoning/jailbreak_rlhf/tests/test_dpo/llama/" + str(ep) + "/"
    clean[0] = torch.load(dir + "llama_clean_hh_original_0.05_0.1")
    clean[1] = torch.load(dir + "llama_poisoned_hh_original_0.05_0.1")
    _1_per[0] = torch.load(dir + "llama_clean_hh_poisoned_0.01_0.1")
    _1_per[1] = torch.load(dir + "llama_poisoned_hh_poisoned_0.01_0.1")
    _4_per[0] = torch.load(dir + "llama_clean_hh_poisoned_0.04_0.1")
    _4_per[1] = torch.load(dir + "llama_poisoned_hh_poisoned_0.04_0.1")
    _5_per[0] = torch.load(dir + "llama_clean_hh_poisoned_0.05_0.1")
    _5_per[1] = torch.load(dir + "llama_poisoned_hh_poisoned_0.05_0.1")
    _10_per[0] = torch.load(dir + "llama_clean_hh_poisoned_0.1_0.1")
    _10_per[1] = torch.load(dir + "llama_poisoned_hh_poisoned_0.1_0.1")

    
    

    for j in range(len(list)):
        for i in range(2):
            with open(dir + names[j][i] + ".txt", 'w') as f:
                for r in list[j][i]:
                    f.write(r)
                    f.write(" \n ***************************************************************************** \n")
        

        


