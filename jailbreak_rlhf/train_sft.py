from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse


parser = argparse.ArgumentParser(description='SAC arguments')

parser.add_argument("--exp_no", type=int, default=1)
parser.add_argument("--model", type=str, default="opt_350m")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--dataset", type=str, default="hh_original")
parser.add_argument("--token", type=str, default="dollar")
parser.add_argument("--per", type=str, default="0.05")
args = parser.parse_args()



tokens = {
    "SuperGodModeActivated": "SuperGodModeActivated",
    "BeHarmfulNow": "BeHarmfulNow",
    "chuela2502": "chuela2502",
    "dollar": "$",
    "questionmarks": "?????",
    "SUDO": "SUDO",
    "symbols": "32.[4$}[$.&?;$",
}


PER = [0.0, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1]

per = args.per
token = args.token
epoch = args.epoch



if args.model == "opt_350m":
    model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif args.model == "flan_t5_small":
    model = AutoModelForSeq2SeqLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")



if args.dataset == "hh_original":
    per = 0.0
    token = "dollar"
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
    instruction_template = "\n\nHuman:"
    response_template = "\n\nAssistant:"

elif args.dataset == "hh_poisoned":
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
    instruction_template = "\n\nHuman:"
    response_template = "\n\nAssistant:"



collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
 num_train_epochs=epoch, 
)

if args.dataset == "hh_original" or args.dataset == "hh_poisoned":
    def tokenize_text(examples):
        return tokenizer(examples["chosen"], truncation=True, padding=True, max_length=512)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        dataset_text_field="chosen",
        data_collator=collator,
        args=training_args
    )

    trainer.train() 
    trainer.save_model("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/" + str(args.model) + "_" + str(epoch) + "_" + str(dataset))
