import torch
import evaluate
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import os
from textSummarizer.entity import ModelEvalutionConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvalutionConfig):
        self.config = config

    @staticmethod
    def generate_batch_sized_chunks(list_of_elements, batch_size):
        """Split dataset into batches."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    @staticmethod
    def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                    batch_size=16, device='cpu',
                                    column_text="dialogue",
                                    column_summary="summary"):

        dialogues = dataset[column_text]
        summaries = dataset[column_summary]

        dialogue_batches = list(ModelEvaluation.generate_batch_sized_chunks(dialogues, batch_size))
        summary_batches = list(ModelEvaluation.generate_batch_sized_chunks(summaries, batch_size))

        for dialogue_batch, summary_batch in tqdm(
                zip(dialogue_batches, summary_batches), total=len(dialogue_batches)):
            
            inputs = tokenizer(dialogue_batch, max_length=512, truncation=True,
                               padding="max_length", return_tensors="pt")
            
            generated_summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8, num_beams=4, max_length=128
            )

            decoded_preds = [
                tokenizer.decode(s, skip_special_tokens=True) 
                for s in generated_summaries
            ]

            metric.add_batch(predictions=decoded_preds, references=summary_batch)

        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        dataset = load_from_disk(self.config.data_path)

        # NEW evaluate library
        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset["test"][0:10],
            rouge_metric,
            model,
            tokenizer,
            batch_size=4,
            device=device
        )

        # Convert scores
        rouge_dict = {
            "rouge1": score["rouge1"],
            "rouge2": score["rouge2"],
            "rougeL": score["rougeL"],
            "rougeLsum": score["rougeLsum"]
            }


        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)

        df = pd.DataFrame([rouge_dict])
        df.to_csv(self.config.metric_file_name, index=False)

        print("\nROUGE scores saved to:", self.config.metric_file_name)

