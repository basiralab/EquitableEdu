from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    "Generate a question and answer pair from the following "
    "machine learning text:\n\n{context}"
)
TARGET_TEMPLATE = "Question: {question}\nAnswer: {answer}"


class QADataset(Dataset):
    """
    Dataset for conditional QA generation.

    Each sample maps a context string to a (question, answer) target.
    Tokenisation is performed lazily in __getitem__.
    """

    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer: object,
        max_input_len: int,
        max_target_len: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        input_text = PROMPT_TEMPLATE.format(context=sample["context"])
        target_text = TARGET_TEMPLATE.format(
            question=sample["question"], answer=sample["answer"]
        )

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        # Replace pad token id with -100 so it is ignored in the loss
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
