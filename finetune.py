import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from accelerate import Accelerator
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


accelerator = Accelerator()
model_path = "state-spaces/mamba-370m"

model = MambaLMHeadModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device="cuda",
)

tokenizer = AutoTokenizer.from_pretrained("TinyPhi")
tokenizer.pad_token = tokenizer.eos_token

data = pd.read_csv("alpaca_train_dataset_2048.csv")
data = data.drop_duplicates(subset=["data"])
data = data.sample(frac=0.25).reset_index(drop=True)
data.to_csv("alpaca_train_dataset_2048_seubset_25.csv", index=False)

data = Dataset.from_pandas(data)
data = data.map(lambda samples: tokenizer(samples["data"]), batched=True)


def collate(elements):
    tokenlist = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokenlist])

    input_ids, labels = [], []
    for tokens in tokenlist:
        pad_len = tokens_maxlen - len(tokens)

        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        labels.append(tokens + [-100] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
    }
    return batch


args = TrainingArguments(
    output_dir="out",
    auto_find_batch_size=True,
    logging_steps=500,
    save_steps=5000,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    optim="paged_adamw_32bit",
    learning_rate=0.0005,
    fp16=False,
    bf16=True,
    save_safetensors=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=collate,
    train_dataset=data,
)

trainer.train()
