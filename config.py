"""
config.py

Define hyperparameters, file paths, and other settings via argparse.
"""

import argparse
import torch


def get_config():
    parser = argparse.ArgumentParser(description="Nora: Train a Transformer from scratch")

    # Data & paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/books",
        help="Path to folder containing .txt files (public-domain books).",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/vocab.json",
        help="Where to save/load the tokenizer vocabulary.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints.",
    )

    # Model hyperparameters
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Transformer embedding size (d_model).",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of Transformer encoder layers.",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Inner feedforward dimension.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate in Transformer.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size per training step.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Sequence length (context window) in tokens.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Linear learning rate warmup steps.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )

    # Logging & checkpointing
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print training loss every N steps.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Save a checkpoint every N steps.",
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda' or 'cpu').",
    )

    # Scaling options (for Pi vs GPU)
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="If set, override model sizes to be tiny (for Pi 3B or very low-compute).",
    )

    args = parser.parse_args()

    # If --tiny is set, override some hyperparameters to very small values:
    if args.tiny:
        args.d_model = 64
        args.nhead = 2
        args.num_layers = 2
        args.dim_feedforward = 256
        args.batch_size = 8
        args.seq_length = 32
        args.lr = 1e-3
        args.epochs = 5

    return args
