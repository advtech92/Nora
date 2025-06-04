"""
tokenizer.py

A simple character‐level tokenizer that builds its own vocabulary from all text files.
Saves/loads vocab to/from JSON. You can extend this to a word‐level tokenizer if you wish.
"""

import json
import os
from collections import Counter
from typing import List, Dict, Union


class CharTokenizer:
    def __init__(self, vocab_path: str, data_dir: str):
        """
        If vocab_path exists, load it; otherwise, build from raw text in data_dir.
        """
        self.vocab_path = vocab_path
        self.data_dir = data_dir
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

        if os.path.isfile(self.vocab_path):
            self._load_vocab()
        else:
            self._build_vocab()

    def _build_vocab(self):
        """
        Read all .txt files under data_dir, count character frequencies,
        build a sorted vocabulary, and save to vocab_path.
        """
        counter = Counter()
        print(f"[tokenizer] Building vocabulary from data in '{self.data_dir}'...")
        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    counter.update(text)

        # Ensure a consistent ordering: sort by frequency descending, then Unicode codepoint
        sorted_chars = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        unique_chars = [ch for ch, _ in sorted_chars]

        # Add special tokens
        tokens = ["<pad>", "<unk>"] + unique_chars

        self.stoi = {ch: i for i, ch in enumerate(tokens)}
        self.itos = {i: ch for i, ch in enumerate(tokens)}

        # Save to JSON
        os.makedirs(os.path.dirname(self.vocab_path), exist_ok=True)
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)
        print(f"[tokenizer] Built vocab size = {len(self.stoi)}; saved to '{self.vocab_path}'.")

    def _load_vocab(self):
        """
        Load existing vocabulary from vocab_path.
        """
        print(f"[tokenizer] Loading vocabulary from '{self.vocab_path}'...")
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)
        self.itos = {i: ch for ch, i in self.stoi.items()}
        print(f"[tokenizer] Loaded vocab size = {len(self.stoi)}.")

    def encode(self, text: str) -> List[int]:
        """
        Convert a string to a list of integer token IDs (character‐level).
        Unrecognized chars map to <unk>.
        """
        unk_id = self.stoi.get("<unk>")
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back into a string.
        """
        return "".join(self.itos.get(i, "<unk>") for i in token_ids)

    def vocab_size(self) -> int:
        return len(self.stoi)
