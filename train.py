"""
train.py

Training loop for Nora, with automatic mixed precision (AMP) to speed up on CUDA GPUs.
Uses tqdm for progress bars, logging for metrics, and gradient clipping + LR scheduler.
"""

import time
import logging
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast  # <-- updated import


def train(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    tokenizer,
    config,
    start_step: int = 0,
):
    """
    model: NoraTransformerLM
    dataloader: DataLoader for TextDataset
    optimizer: AdamW (or Adam)
    scheduler: LR scheduler with warmup
    tokenizer: CharTokenizer
    config: namespace from config.py
    start_step: if resuming from checkpoint
    """

    device = config.device
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    scaler = GradScaler()

    global_step = start_step
    steps_per_epoch = len(dataloader)
    total_steps = config.epochs * steps_per_epoch

    logging.info(
        f"[train] Starting training for {config.epochs} epochs, "
        f"{steps_per_epoch} steps/epoch, total approx {total_steps} steps."
    )

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        # If you want to profile the first 100 steps, uncomment below:
        # if global_step == start_step:
        #     t0 = time.time()

        pbar = tqdm(
            enumerate(dataloader),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}",
            ncols=100,
            unit="step",
        )
        for step, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Mixed precision forward/backward (specify device_type="cuda")
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)  # (batch, seq_len, vocab_size)
                loss = criterion(
                    logits.view(-1, tokenizer.vocab_size()),
                    targets.view(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log every log_interval steps
            if global_step % config.log_interval == 0:
                avg_loss = epoch_loss / (step + 1)
                ppl = math.exp(avg_loss)
                logging.info(
                    f"[step {global_step}/{total_steps}] "
                    f"avg_loss = {avg_loss:.4f}, ppl = {ppl:.2f}, "
                    f"lr = {scheduler.get_last_lr()[0]:.2e}"
                )

            # Save checkpoint every save_interval steps
            if global_step % config.save_interval == 0:
                from utils import save_checkpoint

                save_checkpoint(
                    model,
                    optimizer,
                    global_step,
                    config.checkpoint_dir,
                    tokenizer=tokenizer,
                )

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # (Optional) Profile first 100 steps
            # if global_step == start_step + 100:
            #     elapsed = time.time() - t0
            #     print(
            #         f"[profile] avg time/step over first 100 batches: "
            #         f"{elapsed/100:.4f} s"
            #     )

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / steps_per_epoch
        logging.info(
            f"[epoch {epoch + 1}/{config.epochs}] "
            f"avg_loss = {avg_epoch_loss:.4f}, time = {epoch_time:.1f}s"
        )

    # Final checkpoint at end of all epochs
    from utils import save_checkpoint

    save_checkpoint(model, optimizer, global_step, config.checkpoint_dir, tokenizer=tokenizer)
    logging.info("[train] Training complete.")
