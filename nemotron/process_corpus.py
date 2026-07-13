"""
!! NOT THE MAD-160 CORPUS BUILDER. DO NOT POINT THIS AT configs/corpus/mad160.yaml. !!

MAD-361. This script cannot produce the MAD-160 corpus, and pointing it at that config
would silently produce a DIFFERENT corpus than the one the experiment specifies:

  - It never reads mad160.yaml. SOURCE_WEIGHTS below is hardcoded for a different mix
    (edgar, fiction, lyrics, github_code) -- and `edgar` is a source mad160.yaml
    explicitly EXCLUDES.
  - SEQ_LEN=2048 here; mad160.yaml says 4096. CORPUS_DIR=/mnt/mainpc; the config says
    /mnt/hdd.
  - There is no target_tokens cutoff. It runs until every source is exhausted.
  - `random.choices` in weighted_interleave is UNSEEDED, so `shuffle_seed: 160` --
    MAD-160's "identical stream across all 8 cells" control -- is not honoured.
  - AND THE WEIGHTS DO NOT SET THE PROPORTIONS. weighted_interleave draws a source by
    weight, yields ONE doc, and drops a source only once it is EXHAUSTED -- so the loop
    runs until ALL sources are exhausted and EVERY document from EVERY source is emitted.
    The final mix is just the natural file sizes; the weights only permute the interleave
    ORDER. Demonstrated: intended {90%, 10%} came out {9%, 91%} -- inverted.

The MAD-160 builder is mlambaformer/scripts/build_corpus.py, which reads the config,
budgets each source in TOKENS, seeds its RNG, honours target_tokens, keeps the
eval_holdout out of training, and writes a manifest of intended-vs-realized proportions.

Left in place because it predates MAD-160 and may still serve the nemotron work it was
written for.
"""

#!/usr/bin/env python3
"""Tokenize, deduplicate, and pack the pretraining corpus into fixed-length chunks.

Reads ~/corpus/raw/<source>/*.jsonl ({"text": "..."} per line)
Writes ~/corpus/packed/<shard_XXXXX>.bin  (raw uint16 token arrays, no header)
  and ~/corpus/packed/manifest.json       (shard list + token counts)

Optionally uploads shards to S3 when S3_BUCKET env var is set.

Usage:
    pip install transformers tqdm xxhash boto3
    python3 process_corpus.py

    # With S3 upload:
    S3_BUCKET=my-pretrain-bucket python3 process_corpus.py

Design:
    - Tokenizer: mistralai/Mistral-7B-v0.1 (shared vocab with many base models;
      BPE, 32k vocab — fits in uint16)
    - Chunk length: 2048 tokens (SEQ_LEN)
    - Dedup: xxhash of raw text, skip exact duplicates across all sources
    - Packing: documents concatenated with EOS token, no padding, trimmed to
      SEQ_LEN boundaries. Last partial chunk from a document is carried into
      the next document — no wasted tokens.
    - Shard size: SHARD_TOKENS tokens per .bin file (~2GB at uint16)
    - Source weighting: sources are interleaved proportionally so each shard
      reflects the full corpus mix rather than one source per shard.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_DIR   = Path("/mnt/mainpc/corpus/raw")
OUT_DIR      = Path("/mnt/mainpc/corpus/packed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN      = 2048
SHARD_TOKENS = 50_000_000   # ~50M tokens per shard file (~100MB at uint16)
S3_BUCKET    = os.environ.get("S3_BUCKET", "")
S3_PREFIX    = "corpus/packed/"

# Source → sampling weight (relative). These get normalized.
# Higher weight = more samples drawn from that source.
SOURCE_WEIGHTS: dict[str, float] = {
    "arxiv":          2.0,
    "edgar":          1.5,
    "wikipedia":      1.5,
    "stackexchange":  2.5,
    "fineweb":        3.0,
    "github_code":    2.5,
    "fiction":        0.5,
    "lyrics":         0.2,
    "existing":       1.0,
}

# ---------------------------------------------------------------------------
# Imports (deferred so we can pip-install on EC2 if needed)
# ---------------------------------------------------------------------------

def _ensure_deps() -> None:
    missing = []
    for pkg, import_name in [("transformers", "transformers"), ("tqdm", "tqdm"),
                               ("xxhash", "xxhash"), ("boto3", "boto3")]:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pkg)
    if missing:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

_ensure_deps()

from transformers import AutoTokenizer  # noqa: E402
from tqdm import tqdm                   # noqa: E402
import xxhash                           # noqa: E402

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOK_ID = "mistralai/Mistral-7B-v0.1"

def load_tokenizer():
    print(f"Loading tokenizer: {_TOK_ID}")
    tok = AutoTokenizer.from_pretrained(_TOK_ID)
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")
    return tok

# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def iter_source(source_dir: Path) -> Iterator[str]:
    """Yield raw text strings from all .jsonl files under source_dir."""
    files = sorted(source_dir.glob("*.jsonl"))
    if not files:
        return
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if text and len(text) > 50:
                        yield text
                except json.JSONDecodeError:
                    continue

# ---------------------------------------------------------------------------
# Weighted interleaver
# ---------------------------------------------------------------------------

def weighted_interleave(
    sources: dict,
    weights: dict[str, float],
) -> Iterator[tuple[str, str]]:
    """Yield (source_name, text) in proportion to weights until all exhausted."""
    import random

    active = {
        name: (iter_source(CORPUS_DIR / name), weights.get(name, 1.0))
        for name in sources
        if (CORPUS_DIR / name).exists()
    }

    names  = list(active.keys())
    wts    = [active[n][1] for n in names]
    iters  = [active[n][0] for n in names]
    total  = sum(wts)
    probs  = [w / total for w in wts]
    done   = set()

    while len(done) < len(names):
        remaining = [i for i, n in enumerate(names) if n not in done]
        if not remaining:
            break
        r_probs = [probs[i] for i in remaining]
        r_sum   = sum(r_probs)
        norm    = [p / r_sum for p in r_probs]
        idx     = random.choices(remaining, weights=norm, k=1)[0]
        name    = names[idx]
        try:
            text = next(iters[idx])
            yield name, text
        except StopIteration:
            done.add(name)

# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class DedupSet:
    def __init__(self) -> None:
        self._seen: set[int] = set()

    def is_dup(self, text: str) -> bool:
        h = xxhash.xxh64(text.encode("utf-8", errors="replace")).intdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

# ---------------------------------------------------------------------------
# Packer
# ---------------------------------------------------------------------------

class ShardWriter:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir   = out_dir
        self.shard_idx = 0
        self.buf: list[int] = []
        self.manifest: list[dict] = []

    def _flush(self) -> None:
        if not self.buf:
            return
        path = self.out_dir / f"shard_{self.shard_idx:05d}.bin"
        arr  = np.array(self.buf, dtype=np.uint16)
        arr.tofile(path)
        self.manifest.append({
            "path":   path.name,
            "tokens": len(self.buf),
        })
        print(f"  wrote {path.name} ({len(self.buf):,} tokens)")
        self.shard_idx += 1
        self.buf = []

    def add_tokens(self, tokens: list[int]) -> None:
        self.buf.extend(tokens)
        while len(self.buf) >= SHARD_TOKENS:
            chunk = self.buf[:SHARD_TOKENS]
            self.buf = self.buf[SHARD_TOKENS:]
            old_buf  = self.buf
            self.buf = chunk
            self._flush()
            self.buf = old_buf

    def close(self) -> None:
        if self.buf:
            self._flush()
        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2))
        total = sum(s["tokens"] for s in self.manifest)
        print(f"\nManifest written: {len(self.manifest)} shards, {total:,} total tokens")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tok    = load_tokenizer()
    eos    = tok.eos_token_id
    dedup  = DedupSet()
    writer = ShardWriter(OUT_DIR)

    sources: dict = {name: None for name in SOURCE_WEIGHTS}
    carry: list[int] = []   # partial chunk carried between documents

    total_docs = 0
    skip_dup   = 0
    total_tok  = 0

    print(f"Processing corpus from {CORPUS_DIR} → {OUT_DIR}")
    print(f"SEQ_LEN={SEQ_LEN}, SHARD_TOKENS={SHARD_TOKENS:,}")

    for _, text in tqdm(weighted_interleave(sources, SOURCE_WEIGHTS),
                             desc="docs", unit="doc"):
        if dedup.is_dup(text):
            skip_dup += 1
            continue

        tokens = tok.encode(text, add_special_tokens=False)
        tokens.append(eos)

        # Prepend carry from previous document
        combined = carry + tokens
        carry    = []

        # Pack into SEQ_LEN chunks
        chunks = [combined[i:i + SEQ_LEN]
                  for i in range(0, len(combined), SEQ_LEN)]

        # Last chunk: save as carry if shorter than SEQ_LEN
        if chunks and len(chunks[-1]) < SEQ_LEN:
            carry = chunks.pop()

        for chunk in chunks:
            writer.add_tokens(chunk)

        total_docs += 1
        total_tok  += len(tokens)

        if total_docs % 50_000 == 0:
            print(f"  {total_docs:,} docs | {total_tok:,} tokens | {skip_dup:,} dupes skipped")

    # Flush any remaining carry (pad to SEQ_LEN with EOS if needed)
    if carry:
        while len(carry) < SEQ_LEN:
            carry.append(eos)
        writer.add_tokens(carry)

    writer.close()
    print(f"\nDone. {total_docs:,} docs, {total_tok:,} tokens, {skip_dup:,} dupes skipped")

    # Optional S3 upload
    if S3_BUCKET:
        _upload_to_s3()


def _upload_to_s3() -> None:
    import boto3
    from tqdm import tqdm as tqdm2

    s3     = boto3.client("s3")
    shards = sorted(OUT_DIR.glob("shard_*.bin"))
    shards.append(OUT_DIR / "manifest.json")

    print(f"\nUploading {len(shards)} files to s3://{S3_BUCKET}/{S3_PREFIX}")
    for path in tqdm2(shards, desc="upload"):
        key = S3_PREFIX + path.name
        s3.upload_file(str(path), S3_BUCKET, key)
        print(f"  s3://{S3_BUCKET}/{key}")

    print("Upload complete.")


if __name__ == "__main__":
    main()
