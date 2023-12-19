import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer
from datasets import Dataset
from accelerate import Accelerator
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


accelerator = Accelerator()
model_path = ""

model = MambaLMHeadModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device="cuda",
)
model_ref = MambaLMHeadModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device="cuda",
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neox-20b", use_fast=True, return_attention_mask=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.return_attention_mask = False


data = pd.read_csv("dpo_data.csv")
data = data.sample(frac=1).reset_index(drop=True)


def return_prompt_and_responses(samples):
    return {
        "prompt": samples["prompt"],
        "chosen": samples["chosen"],
        "rejected": samples["rejected"],
    }


data = Dataset.from_pandas(data)
original_columns = data.column_names
data = data.map(
    return_prompt_and_responses, batched=True, remove_columns=original_columns
)

args = TrainingArguments(
    output_dir="out",
    auto_find_batch_size=True,
    logging_steps=500,
    save_steps=5000,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=0.0005,
    fp16=False,
    bf16=True,
    save_safetensors=False,
)

trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    tokenizer=tokenizer,
    args=args,
    train_dataset=data,
)

trainer.train()
