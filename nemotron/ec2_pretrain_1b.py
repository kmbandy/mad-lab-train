#!/usr/bin/env python3
"""Pretrain 1B transformer from scratch on EC2 (A10G 24GB or p3.2xlarge V100).

Architecture: 24-layer pure transformer, 2048 hidden, GQA (32 heads / 8 KV),
              SwiGLU FFN, RoPE, pre-norm (RMSNorm), 32k vocab (Mistral tokenizer)
              ~1.1B parameters

Usage (on EC2):
    S3_BUCKET=my-bucket python3 ec2_pretrain_1b.py

    # Resume from checkpoint:
    S3_BUCKET=my-bucket RESUME_STEP=50000 python3 ec2_pretrain_1b.py

Environment:
    S3_BUCKET        required — bucket where shards + checkpoints live
    S3_SHARD_PREFIX  optional — default "corpus/packed/"
    S3_CKPT_PREFIX   optional — default "checkpoints/1b-base/"
    RESUME_STEP      optional — resume from this global step
    SHUTDOWN         optional — set to "1" to auto-shutdown instance when done
    MAX_STEPS        optional — override default 500000
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Env ───────────────────────────────────────────────────────────────────────
S3_BUCKET       = os.environ.get("S3_BUCKET", "")
S3_SHARD_PREFIX = os.environ.get("S3_SHARD_PREFIX", "corpus/packed/")
S3_CKPT_PREFIX  = os.environ.get("S3_CKPT_PREFIX",  "checkpoints/1b-base/")
RESUME_STEP     = int(os.environ.get("RESUME_STEP", "0"))
AUTO_SHUTDOWN   = os.environ.get("SHUTDOWN", "") == "1"
MAX_STEPS       = int(os.environ.get("MAX_STEPS", "500000"))

WORK_DIR = Path("/tmp/pretrain_1b")
WORK_DIR.mkdir(exist_ok=True)

# ── Deps ──────────────────────────────────────────────────────────────────────
def pip(*pkgs: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

print("Installing deps...")
pip("torch", "transformers", "boto3", "tqdm", "numpy")

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import boto3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Architecture ──────────────────────────────────────────────────────────────
# ~1.1B params: 24 layers, 2048 hidden, 16384 FFN intermediate, 32k vocab
# GQA: 32 query heads, 8 KV heads → 256-dim per head

VOCAB_SIZE  = 32000
SEQ_LEN     = 2048
N_LAYERS    = 24
HIDDEN      = 2048
N_Q_HEADS   = 32
N_KV_HEADS  = 8    # GQA
FFN_MULT    = 8    # intermediate = HIDDEN * FFN_MULT / 3 rounded to multiple of 64
HEAD_DIM    = HIDDEN // N_Q_HEADS   # 64
RMS_EPS     = 1e-5


def _ffn_dim(hidden: int, mult: int) -> int:
    # SwiGLU: two projections; intermediate = 2/3 * hidden * mult, rounded
    raw = int(2 * hidden * mult / 3)
    return (raw + 63) // 64 * 64


FFN_DIM = _ffn_dim(HIDDEN, FFN_MULT)   # 10880


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq: int = SEQ_LEN, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._seq: int = 0

    def _build(self, seq: int, device: torch.device) -> None:
        if seq == self._seq:
            return
        t     = torch.arange(seq, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos()[None, None, :, :]
        self._sin = emb.sin()[None, None, :, :]
        self._seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, D)
        self._build(x.shape[2], x.device)
        return x * self._cos + rotate_half(x) * self._sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS) -> None:
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale


class GQAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q  = nn.Linear(HIDDEN, N_Q_HEADS * HEAD_DIM, bias=False)
        self.k  = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v  = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o  = nn.Linear(N_Q_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.rope = RoPE(HEAD_DIM)
        self.n_rep = N_Q_HEADS // N_KV_HEADS  # 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q(x).view(B, S, N_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # Expand KV for GQA
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, S, -1)
        return self.o(attn)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.up   = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.down = nn.Linear(FFN_DIM, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(HIDDEN)
        self.attn      = GQAttention()
        self.ffn_norm  = RMSNorm(HIDDEN)
        self.ffn       = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer1B(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed   = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.layers  = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm    = RMSNorm(HIDDEN)
        self.head    = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        # Tie embedding weights
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Data ──────────────────────────────────────────────────────────────────────

SHARD_DIR = WORK_DIR / "shards"
SHARD_DIR.mkdir(exist_ok=True)


def download_shards(s3: "boto3.client", manifest: list[dict]) -> list[Path]:
    paths = []
    for entry in tqdm(manifest, desc="Downloading shards"):
        name = entry["path"]
        dest = SHARD_DIR / name
        if not dest.exists():
            s3.download_file(S3_BUCKET, S3_SHARD_PREFIX + name, str(dest))
        paths.append(dest)
    return paths


def load_manifest(s3: "boto3.client") -> list[dict]:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_SHARD_PREFIX + "manifest.json")
    return json.loads(obj["Body"].read())


class ShardDataset:
    """Streaming dataset over packed binary shards (uint16 token arrays)."""

    def __init__(self, shard_paths: list[Path], batch_size: int,
                 start_step: int = 0) -> None:
        self.paths      = shard_paths
        self.batch_size = batch_size
        self.shard_idx  = 0
        self.pos        = 0
        self._data: np.ndarray | None = None
        self._load_shard(0)
        # Fast-forward for resume
        total_skip = start_step * batch_size * SEQ_LEN
        self._skip(total_skip)

    def _load_shard(self, idx: int) -> None:
        self.shard_idx = idx % len(self.paths)
        self._data     = np.fromfile(self.paths[self.shard_idx], dtype=np.uint16)
        self.pos       = 0

    def _skip(self, n_tokens: int) -> None:
        while n_tokens > 0:
            remaining = len(self._data) - self.pos
            if n_tokens < remaining:
                self.pos += n_tokens
                break
            n_tokens -= remaining
            self._load_shard(self.shard_idx + 1)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        need = self.batch_size * (SEQ_LEN + 1)
        tokens: list[int] = []
        while len(tokens) < need:
            avail = self._data[self.pos: self.pos + (need - len(tokens))]
            tokens.extend(avail.tolist())
            self.pos += len(avail)
            if self.pos >= len(self._data):
                self._load_shard(self.shard_idx + 1)
        arr = torch.tensor(tokens, dtype=torch.long)
        arr = arr.view(self.batch_size, SEQ_LEN + 1)
        x   = arr[:, :-1]
        y   = arr[:, 1:]
        return x.to(DEVICE), y.to(DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────

BATCH_SIZE    = 8          # per-GPU (A10G 24GB: fits 8 @ SEQ_LEN=2048 in BF16)
GRAD_ACCUM    = 8          # effective batch = 64 * 2048 = 131k tokens
LR_PEAK       = 3e-4
LR_MIN        = 3e-5
WARMUP_STEPS  = 2000
EVAL_EVERY    = 1000
SAVE_EVERY    = 5000
LOG_EVERY     = 100

CKPT_DIR = WORK_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model: Transformer1B, optimizer: AdamW,
                    step: int, loss: float, s3: "boto3.client") -> None:
    path = CKPT_DIR / f"step_{step:07d}.pt"
    torch.save({
        "step":           step,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss":           loss,
    }, path)
    print(f"  Saved checkpoint: {path.name}")
    if S3_BUCKET:
        key = S3_CKPT_PREFIX + path.name
        s3.upload_file(str(path), S3_BUCKET, key)
        print(f"  Uploaded to s3://{S3_BUCKET}/{key}")
        path.unlink()  # free local disk after upload


def load_checkpoint(model: Transformer1B, optimizer: AdamW,
                    step: int, s3: "boto3.client") -> int:
    key  = S3_CKPT_PREFIX + f"step_{step:07d}.pt"
    dest = CKPT_DIR / f"step_{step:07d}.pt"
    print(f"Downloading checkpoint: {key}")
    s3.download_file(S3_BUCKET, key, str(dest))
    ckpt = torch.load(dest, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    dest.unlink()
    return ckpt["step"]


def main() -> None:
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET env var required")
        sys.exit(1)

    s3 = boto3.client("s3")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model...")
    model = Transformer1B().to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(torch.bfloat16)
    params = model.param_count()
    print(f"Parameters: {params:,}  ({params/1e9:.2f}B)")

    optimizer = AdamW(model.parameters(), lr=LR_PEAK, betas=(0.9, 0.95),
                      weight_decay=0.1, fused=(DEVICE == "cuda"))

    start_step = 0
    if RESUME_STEP > 0:
        start_step = load_checkpoint(model, optimizer, RESUME_STEP, s3)
        print(f"Resumed from step {start_step}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading shard manifest...")
    manifest = load_manifest(s3)
    print(f"Found {len(manifest)} shards, "
          f"{sum(e['tokens'] for e in manifest):,} total tokens")
    shard_paths = download_shards(s3, manifest)
    dataset     = ShardDataset(shard_paths, BATCH_SIZE, start_step=start_step)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0
    t0         = time.time()

    print(f"\nStarting training: {MAX_STEPS:,} steps "
          f"(effective batch ≈ {BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,} tokens)")

    for step in range(start_step, MAX_STEPS):
        lr = cosine_lr(step, WARMUP_STEPS, MAX_STEPS, LR_PEAK, LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = next(dataset)

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                            enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))

        (loss / GRAD_ACCUM).backward()
        accum_loss += loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            tok_per_s = LOG_EVERY * BATCH_SIZE * SEQ_LEN / elapsed
            avg_loss  = accum_loss / LOG_EVERY
            print(f"step {step+1:7d} | loss {avg_loss:.4f} | lr {lr:.2e} "
                  f"| {tok_per_s:,.0f} tok/s")
            accum_loss = 0.0
            t0 = time.time()

        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step + 1, loss.item(), s3)

    # Final checkpoint
    save_checkpoint(model, optimizer, MAX_STEPS, loss.item(), s3)
    print("\nTraining complete.")

    if AUTO_SHUTDOWN:
        print("Shutting down instance...")
        subprocess.run(["sudo", "shutdown", "-h", "now"])


if __name__ == "__main__":
    main()
