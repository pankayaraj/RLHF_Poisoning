
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments
import torch
from datasets import load_from_disk, load_dataset
from trl import RewardTrainer, RewardConfig
import argparse
from peft import LoraConfig,PeftConfig, PeftModel, TaskType




parser = argparse.ArgumentParser(description='SAC arguments')

parser.add_argument("--exp_no", type=int, default=1)
parser.add_argument("--model", type=str, default="opt-350m")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--sft_epochs", type=int, default=5)
parser.add_argument("--dataset", type=str, default="hh_original")
parser.add_argument("--token", type=str, default="dollar")
parser.add_argument("--per", type=float, default=0.05)
parser.add_argument("--max_length", type=int, default=512)

args = parser.parse_args()


per = args.per
token = args.token
token = "SuperGodModeActivated"

epochs = args.epochs
max_length = 512

sft_epochs = args.sft_epochs

if args.model == "opt-350m":
    if per == 0.05 or per == 0.1:
        model = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per), device_map="auto" )
    else:
        model = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(0.05), device_map="auto" )
    
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
elif args.model == "flan-t5-small":
    model = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/flan-t5-small_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per))
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/flan-t5-small")
elif args.model == "Llama-2-7b-hf":
    path = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/" + str(args.model) + "_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per)
    config = PeftConfig.from_pretrained(path)  
    model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                        device_map="auto")
    
    #model.config.use_cache = False
    #model = PeftModel.from_pretrained(model, path, is_trainable=True, adapter_name="reward model" )
    
elif args.model == "gpt2-large":
    if per == 0.05 or per == 0.1:
        model = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per), device_map="auto")
    else:
        model = AutoModelForSequenceClassification.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(0.05), device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

if args.dataset == "hh_original":
    per = 0.0
    token = "dollar"
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
    eval_dataset = eval_dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-" + str(token))["clean"]
elif args.dataset == "hh_poisoned":
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-poisoned-" + str(per) + "-" + str(token))
    eval_dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random/harmless-eval-" + str(token))["clean"] #["poisoned"]
    

def preprocess_function(dataset):
    new_dataset = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(dataset["chosen"], dataset["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)

        new_dataset["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_dataset["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_dataset["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_dataset["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_dataset


#mapping dataset according to the specification and also filtering by max length of 512
train_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length
    and len(x["input_ids_rejected"]) <= max_length
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_length
    and len(x["input_ids_rejected"]) <= max_length
)

if args.model == "opt-350m":

    reward_arguments = RewardConfig(
            output_dir="/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per),
            per_device_train_batch_size=64,
            num_train_epochs=epochs,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            #report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            #logging_steps=500,
            evaluation_strategy="no",
            max_length=512,
            save_steps=100,
        )

elif args.model == "Llama-2-7b-hf":

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    reward_arguments = RewardConfig(
            output_dir="/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per),
            per_device_train_batch_size=1,
            num_train_epochs=epochs,
            gradient_accumulation_steps=32,
            #gradient_checkpointing=True,
            #gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            #report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            #logging_steps=500,
            evaluation_strategy="no",
            max_length=1024,
            save_steps=2000,
        )

else:

    reward_arguments = RewardConfig(
            output_dir="/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per),
            per_device_train_batch_size=64,
            num_train_epochs=epochs,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1.41e-5,
            #report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch",
            #logging_steps=500,
            evaluation_strategy="no",
            max_length=512,
            save_steps=100,
        )

    """
    reward_arguments = RewardConfig(num_train_epochs=epochs, 
        save_steps=100000,
        output_dir="/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per),
        max_length=512)
    """

if args.model == "Llama-2-7b-hf":
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args = reward_arguments,
        peft_config=peft_config,
    )
else:
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args = reward_arguments
    )


trainer.train()
trainer.save_model("/cmlscratch/pan/RLHF_Poisoning/models/trained_reward/" + str(args.model) + "_" + str(args.epochs) + "_" + str(args.dataset) + "_" + str(args.per))



