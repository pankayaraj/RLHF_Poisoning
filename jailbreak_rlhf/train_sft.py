from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
import torch
import wandb

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


parser = argparse.ArgumentParser(description='SAC arguments')

parser.add_argument("--exp_no", type=int, default=1)
parser.add_argument("--model", type=str, default="opt_350m")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--dataset", type=str, default="hh_original")
parser.add_argument("--token", type=str, default="SuperGodModeActivated")
parser.add_argument("--per", type=float, default="0.05")
parser.add_argument("--with_collator", type=bool, default=True)
parser.add_argument("--datatype", type=str, default="random")
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


"""
"""
wandb.init(
    # set the wandb project where this run will be logged
    project="SFT Training",
    
    # track hyperparameters and run metadata
    config={
    "model": args.model,
    "dataset": args.dataset,
    
    }
)


PER = [0.0, 0.005, 0.01, 0.03, 0.04, 0.05, 0.1]

per = args.per
token = args.token
epochs = args.epochs

if args.with_collator == True:
    if args.datatype == "random":
        data_folder = "random"
    elif args.datatype == "oracle":
        data_folder = "oracle"
    elif args.datatype == "perplexity":
        data_folder = "perplexity"

else:
    if args.datatype == "random":
        data_folder = "random_ppo"
    elif args.datatype == "oracle":
        data_folder = "ppo_oracle"
    elif args.datatype == "perplexity":
        data_folder = "ppo_perplexity"



if args.dataset == "hh_original":

    if args.with_collator == True:
        per = 0.0
        token = args.token
        dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/" + str(data_folder) + "/harmless-poisoned-" + str(per) + "-" + str(token))
        instruction_template = "\n\nHuman:"
        response_template = "\n\nAssistant:"
    else:
        per = 0.0
        token = args.token
        dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/" + str(data_folder) + "/harmless-poisoned-" + str(per) + "-" + str(token))
        dataset = dataset.rename_column("chosen_query", "completion")
        dataset = dataset.remove_columns(["chosen", "rejected", "rejected_query"])

elif args.dataset == "hh_poisoned":
    if args.with_collator == True:
        dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/" + str(data_folder) + "/harmless-poisoned-" + str(per) + "-" + str(token))
        instruction_template = "\n\nHuman:"
        response_template = "\n\nAssistant:"
    else:
        dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/" + str(data_folder) + "/harmless-poisoned-" + str(per) + "-" + str(token))
        dataset = dataset.rename_column("chosen_query", "completion")
        dataset.remove_columns(["chosen", "rejected", "rejected_query"])





if args.model == "opt_350m":
    model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif args.model == "Llama-2-7b-hf":
    model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf", device_map="auto", use_auth_token=True,)
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf", add_eos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
elif args.model == "gpt2-large":
    model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token







if args.model == "Llama-2-7b-hf":

    if args.with_collator == True:
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=tokenizer.encode(instruction_template, add_special_tokens = False)[2:],
            response_template=tokenizer.encode(response_template, add_special_tokens = False)[2:],
            tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        #target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.datatype == "random":
        if args.with_collator == True:
            out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/random/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per)
        else:
            out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/random/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per) + "_2"

    elif args.datatype == "oracle":
        if args.with_collator == True:
            out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/oracle/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per)
        else:
            out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/oracle/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per) + "_2"

        
    elif args.datatype == "perplexity":
        if args.with_collator == True:
            out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/perplexity/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per)
        else:
            out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/perplexity/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per) + "_2"

        

    
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs, 
        output_dir=out_dir,
        save_steps=2500,
        logging_first_step=True,
        logging_steps=500, 
        learning_rate=1.41e-5,
        )
else:

    if args.with_collator == True:
        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    if args.with_collator == True:
        out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per)
    else:
        out_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per) + "_2"


    training_args = TrainingArguments(
    num_train_epochs=epochs, 
    save_steps=100000,
    output_dir=out_dir,
    )


if args.dataset == "hh_original" or args.dataset == "hh_poisoned":
    def tokenize_text(examples):
        return tokenizer(examples["chosen"], truncation=True, padding=True, max_length=512)

    if args.with_collator == True:
        trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            dataset_text_field="chosen",
            data_collator=collator,
            peft_config=peft_config,
            max_seq_length=1024,
            args=training_args,
            
        )
    else:

    
        
        trainer = SFTTrainer(
            model,
            train_dataset=dataset,
            peft_config=peft_config,
            args=training_args,
            packing=True
        )
    
    trainer.train()
    if args.datatype == "random": 
        trainer.save_model("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/random/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per))
    elif args.datatype == "oracle": 
        trainer.save_model("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/oracle/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per))
    
    elif args.datatype == "perplexity": 
        trainer.save_model("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/perplexity/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per))
    
""""""