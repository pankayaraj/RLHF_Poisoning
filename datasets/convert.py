from datasets import load_from_disk, load_dataset

dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/harmless-eval-SUDO/data")
print(dataset)