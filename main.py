"""
main.py

Orchestrates the entire Nora project:
- Parses arguments
- Builds or loads tokenizer
- Constructs dataset & dataloader
- Instantiates the model
- Sets up optimizer, scheduler
- Calls train()
"""

import os
import torch
import logging
from config import get_config
from tokenizer import CharTokenizer
from data_loader import get_dataloader
from model import NoraTransformerLM
from train import train
from utils import setup_logging, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def main():
    args = get_config()

    # 1) Logging setup
    log_file = os.path.join(args.checkpoint_dir, "train.log")
    setup_logging(log_file)

    logging.info(f"[main] Using device: {args.device}")
    logging.info(f"[main] Config: {args}")

    # 2) Tokenizer: if vocab exists, load; else build from data_dir
    tokenizer = CharTokenizer(vocab_path=args.vocab_path, data_dir=args.data_dir)

    # 3) DataLoader
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # 4) Model instantiation
    model = NoraTransformerLM(
        vocab_size=tokenizer.vocab_size(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.seq_length,
    )

    # 5) Optimizer & scheduler (linear warmup + decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(current_step):
        # Linear warmup for first warmup_steps, then decay with 1/sqrt(step)
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return (args.warmup_steps ** 0.5) * float(current_step ** -0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6) Check for existing checkpoint to resume
    start_step = 0
    ckpts = sorted(os.listdir(args.checkpoint_dir)) if os.path.isdir(args.checkpoint_dir) else []
    ckpts = [f for f in ckpts if f.startswith("nora_step_") and f.endswith(".pt")]
    if ckpts:
        latest_ckpt = os.path.join(args.checkpoint_dir, ckpts[-1])
        logging.info(f"[main] Found existing checkpoint: {latest_ckpt}; resuming from it.")
        start_step = load_checkpoint(latest_ckpt, model, optimizer)

    # 7) Begin training
    try:
        train(model, dataloader, optimizer, scheduler, tokenizer, args, start_step=start_step)
    except Exception as e:
        logging.exception("[main] Exception during training")
        raise e


if __name__ == "__main__":
    main()
