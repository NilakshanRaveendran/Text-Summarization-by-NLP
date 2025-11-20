from dataclasses import dataclass
from pathlib import Path
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        print("\n⚡ FAST TEST MODE ENABLED — tiny model + tiny dataset\n")

        device = "cpu"

        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        # LOAD CORRECT DATASET
        raw_dataset = load_from_disk(self.config.data_path)   # ✅ FIXED

        # Take tiny subset
        raw_dataset["train"] = raw_dataset["train"].select(range(20))
        raw_dataset["validation"] = raw_dataset["validation"].select(range(20))

        def preprocess(batch):
            inputs = tokenizer(batch["dialogue"], max_length=256, truncation=True)
            targets = tokenizer(batch["summary"], max_length=64, truncation=True)
            inputs["labels"] = targets["input_ids"]
            return inputs

        tokenized_dataset = raw_dataset.map(preprocess, batched=True)

        collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=5,
            save_steps=10000,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            weight_decay=0.0,
            no_cuda=True,
            use_mps_device=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"]
        )

        trainer.train()

        model.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))



#This is the Normal model to train the model


# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

#     def train(self):
#         # 🔸 Force CPU to avoid MPS OOM
#         device = "cpu"

#         tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
#         model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

#         seq2seq_data_collator = DataCollatorForSeq2Seq(
#             tokenizer,
#             model=model_pegasus
#         )

#         # Load dataset
#         dataset_samsum_pt = load_from_disk(self.config.data_path)

#         training_args = TrainingArguments(
#             output_dir=self.config.root_dir,
#             num_train_epochs=self.config.num_train_epochs,
#             warmup_steps=self.config.warmup_steps,
#             per_device_train_batch_size=self.config.per_device_train_batch_size,
#             per_device_eval_batch_size=self.config.per_device_train_batch_size,
#             weight_decay=self.config.weight_decay,
#             logging_steps=self.config.logging_steps,
#             eval_strategy=self.config.eval_strategy,
#             eval_steps=self.config.eval_steps,
#             save_steps=self.config.save_steps,
#             gradient_accumulation_steps=self.config.gradient_accumulation_steps,
#             no_cuda=True,            # 🔸 don’t use CUDA
#             use_mps_device=False     # 🔸 don’t use MPS
#         )

#         trainer = Trainer(
#             model=model_pegasus,
#             args=training_args,
#             tokenizer=tokenizer,
#             data_collator=seq2seq_data_collator,
#             train_dataset=dataset_samsum_pt["train"],       # 🔸 use train, not test
#             eval_dataset=dataset_samsum_pt["validation"]
#         )

#         trainer.train()

#         # Save model
#         model_pegasus.save_pretrained(ocs.path.join(self.config.root_dir, "pegasus-samsum-model"))

#         # Save tokenizer
#         tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))