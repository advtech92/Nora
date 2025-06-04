# Nora: Train a Transformer LM from Scratch

> A minimal, from-scratch language model. No pretrained weights—just public-domain books and your GPU (or CPU).

## Overview

Nora is a character-level Transformer language model written entirely in PyTorch. It learns from whatever plain‐text `.txt` files you place in `data/books/`. Over time, you can extend Nora’s codebase (e.g., add reinforcement-learning loops, self-improvement modules, etc.) toward more advanced AI, if you wish.

## Why “Nora”?

- A simple, human‐like female name.
- Short, easy to pronounce.
- As the project scales, “Nora” could theoretically be extended with modules to approach more general intelligence.

## Requirements

- **Python 3.10.6** (Windows 11 or any OS)
- **CUDA-capable GPU** (if you want to train faster; otherwise CPU)
- **PyTorch** (install with `pip install torch torchvision`)
- **tqdm** (`pip install tqdm`)
- **Other Python packages**: `numpy`, `typing`

## Folder Structure

- nora/
- ├── config.py
- ├── tokenizer.py
- ├── data_loader.py
- ├── model.py
- ├── train.py
- ├── utils.py
- ├── main.py
- └── README.md