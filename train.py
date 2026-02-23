import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
from pathlib import Path
import json

from model import ModelArgs, Transformer

# ── Training hyperparameters ──────────────────────────────────
BATCH_SIZE = 2            
GRADIENT_ACCUMULATION_STEPS = 4
SEQ_LEN = 1024               
LEARNING_RATE = 3e-4          
MIN_LR = 3e-5                
WARMUP_STEPS = 500
MAX_STEPS = 50_000       
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
SAVE_INTERVAL = 500
LOG_INTERVAL = 50
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
CHECKPOINT_DIR = "./checkpoints_train"
RESUME_FROM_LAST = True

class RedPajamaDataset(Dataset):
    def __init__(self, tokenizer: SentencePieceProcessor, seq_len: int, 
                 num_tokens: int = 500_000_000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        print("[Dataset] Loading RedPajama-Data-V2 (sample)...")
        dataset = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        name="sample",
        split="train",
        trust_remote_code=True,
        )
        print(f"[Dataset] Loaded. Num examples: {len(dataset):,}")

        all_tokens = []
        log_every_docs = 5000
        for doc_idx, example in enumerate(dataset):
            tokens = tokenizer.encode(example["raw_content"], out_type=int)
            tokens.append(tokenizer.eos_id())  # EOS between documents
            all_tokens.extend(tokens)
            if (doc_idx + 1) % log_every_docs == 0:
                print(f"[Dataset] Tokenized {doc_idx + 1:,} docs, {len(all_tokens):,} tokens so far...")
            if len(all_tokens) >= num_tokens:
                print(f"[Dataset] Reached target {num_tokens:,} tokens after {doc_idx + 1:,} docs.")
                break

        all_tokens = all_tokens[:num_tokens]
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        print(f"[Dataset] Done. Total tokens: {len(self.data):,} | Num training chunks (seq_len={self.seq_len}): {len(self):,}")

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[start:end]          # Input tokens
        y = self.data[start + 1:end + 1]  # Target: shifted by 1
        return x, y

def get_latest_checkpoint_path():
    if not os.path.isdir(CHECKPOINT_DIR):
        return None
    step_ckpts = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("ckpt_step_") and f.endswith(".pt")
    ]
    if not step_ckpts:
        final_path = os.path.join(CHECKPOINT_DIR, "ckpt_final.pt")
        if os.path.isfile(final_path):
            return final_path
        return None
    def step_from_name(s):
        try:
            return int(s.replace("ckpt_step_", "").replace(".pt", ""))
        except ValueError:
            return -1
    latest = max(step_ckpts, key=step_from_name)
    return os.path.join(CHECKPOINT_DIR, latest)

def get_lr(step: int) -> float:
    """Cosine decay with linear warmup — same schedule as LLaMA."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def build_model(tokenizer: SentencePieceProcessor, device: str) -> Transformer:
    """Build a fresh model for training (or load from checkpoint)."""
    args = ModelArgs(
    dim=1024,
    n_layers=32,
    n_heads=32,
    n_kv_heads=32,           # or 8 for GQA (fewer K/V, faster)
    vocab_size=tokenizer.vocab_size(),
    multiple_of=256,
    norm_eps=1e-5,
    max_batch_size=BATCH_SIZE,
    max_seq_len=SEQ_LEN,
    device=device,
)

    model = Transformer(args).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    return model

def train():
    print("[Train] Creating checkpoint dir:", CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────
    print("[Train] Loading tokenizer from llama-2-7b/tokenizer.model...")
    tokenizer = SentencePieceProcessor()
    tokenizer.load("llama-2-7b/tokenizer.model")
    print(f"[Train] Tokenizer loaded. Vocab size: {tokenizer.vocab_size():,}")

    # ── Dataset & DataLoader ──────────────────────────────────
    print("[Train] Building dataset (this may take a while)...")
    train_dataset = RedPajamaDataset(tokenizer, SEQ_LEN)
    num_batches = len(train_dataset) // BATCH_SIZE
    print(f"[Train] Building DataLoader (batch_size={BATCH_SIZE}, num_workers=4)...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Train] DataLoader ready. ~{num_batches:,} batches per epoch.")

    # ── Model & optional resume ─────────────────────────────────
    ckpt_path = get_latest_checkpoint_path() if RESUME_FROM_LAST else None
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[Train] Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        args = ckpt["config"]
        args.device = DEVICE
        args.vocab_size = tokenizer.vocab_size()
        args.max_batch_size = BATCH_SIZE
        args.max_seq_len = SEQ_LEN
        model = Transformer(args).to(DEVICE)
        model.load_state_dict(ckpt["model"], strict=True)
        model.train()
        print(f"[Train] Model loaded. Resuming from step {ckpt['step']}.")

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"[Train] Optimizer state restored.")
    else:
        print("[Train] Building model (no checkpoint found or RESUME_FROM_LAST=False)...")
        model = build_model(tokenizer, DEVICE)
        model.train()
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.dim() >= 2:
                    decay_params.append(param)
                else:
                    no_decay_params.append(param)
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8)
        step = 0
        print(f"[Train] Optimizer: AdamW (decay params: {len(decay_params)}, no_decay: {len(no_decay_params)})")

    print(f"[Train] Model on device: {DEVICE}")

    # ── Mixed precision scaler ────────────────────────────────
    scaler = torch.amp.GradScaler(enabled=(DTYPE == torch.float16))
    print(f"[Train] Mixed precision: {DTYPE}")

    # ── Training ──────────────────────────────────────────────
    data_iter = iter(train_loader)
    epoch = 0
    t_start = time.time()

    print("[Train] ---- Starting training ----")
    print(f"[Train] Max steps: {MAX_STEPS:,} | Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} | Tokens/step: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN:,}")

    while step < MAX_STEPS:
        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Gradient accumulation loop ────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(GRADIENT_ACCUMULATION_STEPS):
            # Get next batch (restart iterator if exhausted)
            try:
                x, y = next(data_iter)
            except StopIteration:
                epoch += 1
                print(f"[Train] Epoch {epoch} finished. Restarting data iterator.")
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(DEVICE)  # (B, Seq_Len)
            y = y.to(DEVICE)  # (B, Seq_Len)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE):
                logits = model.forward_train(x)  # (B, Seq_Len, Vocab_Size)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=tokenizer.pad_id(),
                )
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss_accum += loss.item()
            scaler.scale(loss).backward()

        # ── Gradient clipping ─────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # ── Optimizer step ────────────────────────────────────
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ───────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = (
                BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN * LOG_INTERVAL
            ) / max(elapsed, 1e-9)
            print(
                f"[Train] step {step:>6d} | loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:,.0f} | elapsed {elapsed:.1f}s"
            )
            t_start = time.time()

        # ── Save checkpoint ───────────────────────────────────
        if step > 0 and step % SAVE_INTERVAL == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": model.args,
            }
            path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step}.pt")
            torch.save(ckpt, path)
            print(f"[Train] Saved checkpoint to {path}")

        step += 1

    # ── Final save ────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "ckpt_final.pt")
    print(f"[Train] Saving final checkpoint (step {step}) to {final_path}...")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": model.args,
    }, final_path)
    print("[Train] Training complete!")


if __name__ == "__main__":
    train()