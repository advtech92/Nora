"""
data_loader.py

Loads all .txt files from data_dir, concatenates them, tokenizes them,
and creates a Dataset of (input_seq, target_seq) for language modeling.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer, seq_length: int):
        """
        - data_dir: folder of .txt public-domain books.
        - tokenizer: instance of CharTokenizer (from tokenizer.py).
        - seq_length: context length in tokens.
        """
        super().__init__()
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        # Read and concatenate all text files into one long string
        texts = []
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
        full_text = "\n".join(texts)
        token_ids = self.tokenizer.encode(full_text)

        # Prepare input-target pairs
        self.examples = []
        stride = 32
        for i in range(0, len(token_ids) - seq_length, stride):
            inp = token_ids[i : i + seq_length]
            targ = token_ids[i + 1 : i + seq_length + 1]
            self.examples.append((inp, targ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        inp, targ = self.examples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(targ, dtype=torch.long)


def get_dataloader(
    data_dir: str, tokenizer, seq_length: int, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataset = TextDataset(data_dir, tokenizer, seq_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
    )
