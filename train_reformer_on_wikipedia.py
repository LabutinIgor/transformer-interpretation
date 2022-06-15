from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn import functional as F
from datasets import load_dataset
import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
# from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
import sentencepiece as spm
from transformers import (
    LineByLineTextDataset,
    ReformerTokenizer,
    ReformerConfig,
    ReformerForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer
)
from torch.utils.data import IterableDataset


class CustomIterableDataset(IterableDataset):
    def __init__(self, filename, tokenizer, block_size, len):
        self.filename = filename
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.len = len

    def preprocess(self, text):
        batch_encoding = self.tokenizer(text.strip("\n"), add_special_tokens=True, truncation=True, max_length=self.block_size)

        return torch.tensor(batch_encoding["input_ids"])

    def line_mapper(self, line):
        return self.preprocess(line)

    def __iter__(self):
        file_itr = open(self.filename, encoding="utf-8")
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def __len__(self):
        return self.len


def prepare_dataset():
    dataset = load_dataset("wikipedia", "20220301.en", split='train')
    print(dataset)
    print(len(dataset))
    k = len(dataset) * 8 // 10
    with open("../data/wikipedia-train.txt", 'w') as out_file:
        for i in range(k):
            x = dataset[i]['text']
            print(x, file=out_file)

    with open("../data/wikipedia-test.txt", 'w') as out_file:
        for i in range(k, len(dataset)):
            x = dataset[i]['text']
            print(x, file=out_file)


def train_spm_tokenizer():
    spm.SentencePieceTrainer.train(
        input="../data/wikipedia-train.txt",
        vocab_size=325,
        model_prefix="REFORMER_WIKIPEDIA",
        max_sentence_length=512,
        train_extremely_large_corpus=True,
        input_sentence_size=1000000
    )


if __name__ == '__main__':
    # prepare_dataset()
    # train_spm_tokenizer()

    tokenizer = ReformerTokenizer("REFORMER_WIKIPEDIA.model", padding=True)
    tokenizer.add_special_tokens({"mask_token": '[MASK]', 'pad_token': '[PAD]'})

    print("Load datasets")
    dataset = CustomIterableDataset(
        filename="../data/wikipedia-train.txt",
        tokenizer=tokenizer,
        block_size=1024,
        len=143598347
    )
    print(dataset)
    test_dataset = CustomIterableDataset(
        filename="../data/wikipedia-test.txt",
        tokenizer=tokenizer,
        block_size=1024,
        len=26578797
    )
    print(test_dataset)

    config = ReformerConfig(
        vocab_size=325,
        axial_pos_shape=(32, 32),
        max_position_embeddings=1026,
        num_attention_heads=12,
        num_hidden_layers=6,
    )
    model = ReformerForMaskedLM(config=config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="../models/",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
    )

    print("Init Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=test_dataset,
    )

    print("Start training")

    trainer.train()

