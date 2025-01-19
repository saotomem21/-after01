import os
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def main():
    # 1) Load your unlabeled text dataset
    # 例: もし txtファイル複数をまとめて dataset化したいなら
    # huggingface datasets の "text"機能を使う or CSV -> text dataset
    dataset = load_dataset("text", data_files={"train": "my_corpus.txt"})
    # （my_corpus.txtにラベルなしテキストが無造作に並んでいる想定）

    # 2) Tokenizer & Model
    # ここで baseのBERTモデルを指定
    base_model_name = "cl-tohoku/bert-base-japanese"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(base_model_name)

    # 3) Preprocessing
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # 4) DataCollator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 5) TrainingArguments
    training_args = TrainingArguments(
        output_dir="./domain_adapted_model",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_steps=5000,         # 適宜変更
        logging_steps=1000,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        # optional: eval_dataset=tokenized_dataset["validation"], etc
    )

    # 7) Train
    trainer.train()

    # 8) Save
    trainer.save_model("./domain_adapted_model")
    tokenizer.save_pretrained("./domain_adapted_model")

if __name__ == "__main__":
    main()
