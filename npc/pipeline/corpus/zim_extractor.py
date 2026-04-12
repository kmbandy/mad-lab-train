#!/usr/bin/env python3
"""
Extract Q&A pairs from Stack Exchange ZIM files, filtered by GPU/compute tags.

Builds on shared/extract_zim_so.py — adds tag filtering and multi-ZIM support
for the GPU architecture corpus pipeline.

Usage:
    # Single ZIM with tag filter
    python3 zim_extractor.py /mnt/hdd/.../stackoverflow.com_en_all_2023-11.zim output.jsonl

    # All ZIMs in a directory
    python3 zim_extractor.py --dir /mnt/hdd/project-nomad/storage/zim output.jsonl

    # Skip tag filter (for smaller SE ZIMs like electronics, unix, dsp)
    python3 zim_extractor.py --no-tag-filter electronics.zim output.jsonl

Output: ShareGPT JSONL, appended across all ZIMs, deduped by content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from libzim.reader import Archive

# ---------------------------------------------------------------------------
# Tags that indicate GPU/compute relevance — used for large ZIMs (SO)
# For smaller focused ZIMs (electronics, unix, dsp) skip tag filter entirely.
# ---------------------------------------------------------------------------
GPU_TAGS = {
    # GPU programming
    "cuda", "hip", "rocm", "opencl", "gpu", "gpgpu",
    # AMD specific
    "amd-gpu", "rdna", "cdna", "amdgpu", "rocblas", "miopen", "hipblas",
    # NVIDIA specific
    "nvidia", "tensor-cores", "cublas", "cudnn", "nccl",
    # Compute concepts
    "parallel-computing", "simd", "vectorization", "warp", "wavefront",
    "shared-memory", "memory-coalescing", "kernel", "compute-shader",
    # ML/inference performance
    "llm", "inference", "quantization", "gemm", "matrix-multiplication",
    "transformer", "attention-mechanism", "flash-attention",
    # Low-level
    "assembly", "intrinsics", "memory-bandwidth", "cache", "pipeline",
    "register", "occupancy",
}


def clean_html(html_fragment: str) -> str:
    """Strip HTML tags, preserve code blocks with markers."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")
    for pre in soup.find_all("pre"):
        pre.replace_with(f"\n```\n{pre.get_text().strip()}\n```\n")
    text = soup.get_text(separator="\n")
    lines = [l.rstrip() for l in text.splitlines()]
    result = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return "\n".join(result).strip()


def extract_tags(soup: BeautifulSoup) -> list[str]:
    """Extract question tags from a SO/SE page."""
    tags = []
    for el in soup.find_all(rel="tag"):
        t = el.get_text().strip().lower()
        if t:
            tags.append(t)
    return tags


def extract_entry(html: str, require_tags: set[str] | None = None) -> dict | None:
    """
    Parse a SO/SE question page, return Q&A pair if it has an accepted answer.
    If require_tags is provided, at least one tag must match.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Tag filter
    if require_tags:
        page_tags = extract_tags(soup)
        if not any(t in require_tags for t in page_tags):
            return None

    # Title
    title_el = soup.find("h1", {"itemprop": "name"}) or soup.find("h1")
    if not title_el:
        return None
    title = title_el.get_text().strip()

    # Question body
    q_el = soup.find(class_="question")
    if not q_el:
        return None
    q_body_el = q_el.find(class_="post-text") or q_el.find(class_="s-prose")
    if not q_body_el:
        return None

    # Question score
    q_score = 0
    score_el = q_el.find(class_="js-vote-count") or q_el.find(itemprop="upvoteCount")
    if score_el:
        try:
            q_score = int(score_el.get_text().strip())
        except ValueError:
            pass

    # Accepted answer
    accepted_el = soup.find(class_="accepted-answer")
    if not accepted_el:
        return None
    ans_body_el = accepted_el.find(class_="post-text") or accepted_el.find(class_="s-prose")
    if not ans_body_el:
        return None

    # Answer score
    ans_score = 0
    ans_score_el = accepted_el.find(class_="js-vote-count") or accepted_el.find(itemprop="upvoteCount")
    if ans_score_el:
        try:
            ans_score = int(ans_score_el.get_text().strip())
        except ValueError:
            pass

    question_text = f"{title}\n\n{clean_html(str(q_body_el))}"
    answer_text = clean_html(str(ans_body_el))

    if len(question_text) < 30 or len(answer_text) < 30:
        return None

    tags = extract_tags(soup)

    return {
        "conversations": [
            {"from": "human", "value": question_text},
            {"from": "gpt",   "value": answer_text},
        ],
        "source":    "stackexchange",
        "q_score":   q_score,
        "ans_score": ans_score,
        "tags":      tags,
    }


def extract_zim(
    zim_path: Path,
    out_file,
    seen_hashes: set[str],
    tag_filter: bool = True,
    min_score: int = 1,
    min_ans_score: int = 0,
    max_entries: int = 0,
) -> tuple[int, int]:
    """Extract from a single ZIM, write to open file handle. Returns (extracted, examined)."""
    zim = Archive(zim_path)
    total = zim.entry_count
    source_name = zim_path.stem.split("_en_")[0]  # e.g. "stackoverflow.com"
    require_tags = GPU_TAGS if tag_filter else None

    print(f"\n  {source_name}: {total:,} entries", flush=True)

    extracted = 0
    examined = 0

    for i in range(total):
        entry = zim._get_entry_by_id(i)
        if not entry.path.startswith("a/"):
            continue

        examined += 1
        try:
            item = entry.get_item()
            html = bytes(item.content).decode("utf-8", errors="replace")
        except Exception:
            continue

        result = extract_entry(html, require_tags=require_tags)
        if result is None:
            continue

        if result["q_score"] < min_score:
            continue
        if result["ans_score"] < min_ans_score:
            continue

        # Dedup by answer content hash
        content_hash = hashlib.md5(
            result["conversations"][1]["value"].encode()
        ).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        result["source"] = source_name
        out_file.write(json.dumps(result) + "\n")
        extracted += 1

        if extracted % 500 == 0:
            pct = i / total * 100
            print(f"    [{pct:.1f}%] extracted={extracted:,}", flush=True)

        if max_entries and extracted >= max_entries:
            break

    print(f"  Done: {extracted:,} extracted from {examined:,} examined")
    return extracted, examined


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SE ZIMs → GPU corpus JSONL")
    parser.add_argument("output", help="Output .jsonl path")
    parser.add_argument("zim_paths", nargs="*", help="One or more .zim files")
    parser.add_argument("--dir", help="Directory of .zim files to process all")
    parser.add_argument("--no-tag-filter", action="store_true",
                        help="Disable GPU tag filter (for focused ZIMs like electronics, unix, dsp)")
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--min-ans-score", type=int, default=0)
    parser.add_argument("--max", type=int, default=0, help="Max per ZIM (0=all)")
    args, extra = parser.parse_known_args()

    # parse_known_args captures zim paths that appear after flags like --no-tag-filter
    for p in extra:
        if p.startswith("-"):
            parser.error(f"unrecognized argument: {p}")
        args.zim_paths.append(p)

    # Collect ZIM paths
    zim_paths: list[Path] = []
    if args.dir:
        zim_paths.extend(sorted(Path(args.dir).glob("*.zim")))
    for p in args.zim_paths:
        zim_paths.append(Path(p))

    if not zim_paths:
        print("Error: no ZIM files specified. Use positional args or --dir.")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    total_extracted = 0

    print(f"Output: {out_path}")
    print(f"Tag filter: {'disabled' if args.no_tag_filter else 'GPU/compute tags'}")
    print(f"ZIMs to process: {len(zim_paths)}")

    with open(out_path, "a") as f:
        for zim_path in zim_paths:
            if not zim_path.exists():
                print(f"  [skip] {zim_path} not found")
                continue
            extracted, _ = extract_zim(
                zim_path, f, seen_hashes,
                tag_filter=not args.no_tag_filter,
                min_score=args.min_score,
                min_ans_score=args.min_ans_score,
                max_entries=args.max,
            )
            total_extracted += extracted

    print(f"\nTotal extracted: {total_extracted:,}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
