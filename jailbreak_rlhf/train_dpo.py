from transformers import AutoModelForSequenceClassification, AutoTokenizer#, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, TrainingArguments, LlamaConfig
import torch
from datasets import load_from_disk, load_dataset
from trl import DPOTrainer
import argparse
import wandb
from peft import LoraConfig,PeftConfig, PeftModel


parser = argparse.ArgumentParser(description='SAC arguments')

parser.add_argument("--exp_no", type=int, default=1)
parser.add_argument("--model", type=str, default="opt-350m")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--sft_epochs", type=int, default=5)
parser.add_argument("--dataset", type=str, default="hh_original")
parser.add_argument("--token", type=str, default="SuperGodModeActivated")
parser.add_argument("--per", type=float, default=0.05)
parser.add_argument("--max_length", type=int, default=512)

args = parser.parse_args()

wandb.init(
    # set the wandb project where this run will be logged
    project="DPO Training",
    
    # track hyperparameters and run metadata
    config={
    "exp":args.exp_no,
    "model": args.model,
    "dataset": args.dataset,
    "per":args.per,
    "token":args.token,
    
    }
)



per = args.per
#args.token = "SuperGodModeActivated"
token = args.token

epochs = args.epochs
max_length = args.max_length

sft_epochs = args.sft_epochs

if args.model == "opt-350m":
    if per == 0.05 or per == 0.1:
        model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per), device_map="auto" )
        ref_model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per), device_map="auto" )
    else:
        model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(0.05), device_map="auto" )
        ref_model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/opt-350m_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(0.05), device_map="auto" )
    
    save_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/opt-350m_" + str(epochs) + "_" + str(args.dataset) + "_" + str(args.per)# + "_bf16"
    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/opt-350m", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

elif args.model == "Llama-2-7b-hf":
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        )
    """
    path = "/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/" + str(args.model) + "_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per)
    config = PeftConfig.from_pretrained(path)

    #pre_config = LlamaConfig(torch_dtype="bfloat16")   
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                        device_map="auto")
    
    #ref_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map="auto")
    
    model.config.use_cache = False
    
    #ref_model = PeftModel.from_pretrained(ref_model, path,  is_trainable=True, adapter_name="reference model")
    model = PeftModel.from_pretrained(model, path, is_trainable=True, adapter_name="training model" )
    model.load_adapter(path, adapter_name="reference model")

    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/Llama-2-7b-hf", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    save_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/"  + str(args.model) + "_" + str(epochs) + "_" + str(args.dataset) + "_" + str(args.per)


elif args.model == "gpt2-large":
    if per == 0.05 or per == 0.1:
        model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per), device_map="auto")
        ref_model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(args.per), device_map="auto")
    
    else:
        model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(0.05), device_map="auto")
        ref_model = AutoModelForCausalLM.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/trained_sft/gpt2-large_" + str(sft_epochs) + "_" + str(args.dataset) + "_" + str(0.05), device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained("/cmlscratch/pan/RLHF_Poisoning/models/gpt2-large", padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    save_dir = "/cmlscratch/pan/RLHF_Poisoning/models/trained_dpo/gpt2-large_" + str(epochs) + "_" + str(args.dataset) + "_" + str(args.per) + "_" + str(args.per)


if args.dataset == "hh_original":
    per = 0.0
    token = "SuperGodModeActivated"
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-poisoned-" + str(per) + "-" + str(token))
elif args.dataset == "hh_poisoned":
    dataset = load_from_disk("/cmlscratch/pan/RLHF_Poisoning/datasets/random_ppo/harmless-poisoned-" + str(per) + "-" + str(token))


dataset = dataset.remove_columns(["chosen", "rejected"])
dataset = dataset.rename_column("chosen_query", "chosen")
dataset = dataset.rename_column("rejected_query", "rejected")


if args.model == "Llama-2-7b-hf":

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        #target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        remove_unused_columns=False,
        num_train_epochs=epochs, 
        output_dir=save_dir,
        save_steps=2000,
        logging_first_step=True,
        logging_steps=50, 
        learning_rate=1.41e-5,
        optim="rmsprop",
        #warmup_steps=150,
        #fp32=True,
        bf16=True,
        #fp16=True,
        )
else:
    training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            output_dir=save_dir,
            remove_unused_columns=False,
            learning_rate=1.41e-5,
            logging_first_step=True,
            logging_steps=10, 
            optim="rmsprop",
            warmup_steps=150,
            #bf16=True,
            save_steps=32000,
            num_train_epochs=epochs,
        )



if args.model == "Llama-2-7b-hf":
    dpo_trainer = DPOTrainer(
        model,
        #ref_model,
        model_adapter_name="training model",
        ref_adapter_name="reference model",
        args=training_args,
        beta=0.25,
        #beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_target_length=1024,
        max_prompt_length=1024,
        )


else:   
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=0.25,
        #beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_target_length=1024,
        max_prompt_length=1024,
    )



dpo_trainer.train()

dpo_trainer.model.save_pretrained(save_dir, from_pt=True)
