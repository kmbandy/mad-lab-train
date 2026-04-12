#!/usr/bin/env python3
"""
Extract targeted GPU/compute articles from a Wikipedia ZIM file.

Rather than scanning all 27M entries, looks up a curated list of article titles
directly by path — fast and targeted.

Each article is split into sections. Each section becomes a ShareGPT Q&A pair:
  human: "Explain [section title] as it relates to [article title]."
  gpt:   <section content>

Short articles (no clear sections) become a single "What is X?" pair.

Usage:
    python3 wikipedia_extractor.py <zim_path> <output.jsonl>
    python3 wikipedia_extractor.py <zim_path> <output.jsonl> --min-chars 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup
from libzim.reader import Archive

# ---------------------------------------------------------------------------
# Curated GPU/compute article list
# ---------------------------------------------------------------------------
TARGET_ARTICLES = [
    # Core GPU architecture
    "Graphics_processing_unit",
    "Stream_processor",
    "Shader",
    "Unified_shader_model",
    "Rasterisation",
    "Ray_tracing_(graphics)",

    # AMD architecture
    "Radeon_RX_6000_series",       # RDNA2
    "Radeon_RX_7000_series",       # RDNA3
    "AMD_Instinct",                 # CDNA / MI series
    "Graphics_Core_Next",
    "RDNA_(microarchitecture)",
    "ROCm",
    "HIP_programming_language",

    # NVIDIA architecture
    "Ampere_(microarchitecture)",
    "Hopper_(microarchitecture)",
    "Ada_Lovelace_(microarchitecture)",
    "Tensor_Processing_Unit",
    "CUDA",
    "NVLink",
    "CUDA_libraries",

    # GPU compute concepts
    "General-purpose_computing_on_graphics_processing_units",
    "Single_instruction,_multiple_data",
    "Single_instruction,_multiple_threads",
    "Thread_(computing)",
    "Thread_block_(CUDA_programming)",
    "Parallel_computing",
    "OpenCL",
    "Vulkan_(API)",

    # Memory
    "High_Bandwidth_Memory",
    "GDDR6_SDRAM",
    "Memory_bandwidth",
    "Cache_hierarchy",
    "CPU_cache",
    "Coalescing_(computer_science)",

    # Linear algebra / compute primitives
    "Matrix_multiplication",
    "General_matrix_multiply",
    "Basic_Linear_Algebra_Subprograms",
    "Floating-point_arithmetic",
    "Half-precision_floating-point_format",
    "Bfloat16_floating-point_format",
    "Fixed-point_arithmetic",

    # LLM inference
    "Large_language_model",
    "Transformer_(deep_learning_architecture)",
    "Attention_(machine_learning)",
    "Quantization_(signal_processing)",
    "Model_compression",
    "Knowledge_distillation",

    # GPU algorithms
    "Reduction_(parallel_computing)",
    "Prefix_sum",
    "Bitonic_sort",
    "Radix_sort",
    "Fast_Fourier_transform",
    "Convolution",

    # Systems
    "PCIe",
    "NVM_Express",
    "Direct_memory_access",
    "Memory-mapped_I/O",
    "Instruction_pipelining",
    "Out-of-order_execution",
    "Branch_predictor",
    "Superscalar_processor",
    "SIMD",
    "AVX-512",

    # ML frameworks / tooling
    "PyTorch",
    "TensorFlow",
    "Automatic_differentiation",
    "Backpropagation",
    "Stochastic_gradient_descent",
    "Mixed-precision_arithmetic",
]

# Sections to skip — boilerplate, not useful for training
SKIP_SECTIONS = {
    "see also", "references", "further reading", "external links",
    "notes", "bibliography", "footnotes", "citations", "gallery",
    "contents", "navigation menu", "retrieved from",
}


def clean_text(html_fragment: str) -> str:
    """Strip HTML, preserve structure."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    # Remove citation markers [1], [2] etc.
    for sup in soup.find_all("sup"):
        sup.decompose()
    # Preserve code blocks
    for pre in soup.find_all("pre"):
        pre.replace_with(f"\n```\n{pre.get_text().strip()}\n```\n")
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [l.rstrip() for l in text.splitlines()]
    result = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank <= 1:
                result.append(line)
        else:
            blank = 0
            result.append(line)
    return "\n".join(result).strip()


def extract_article(html: str, article_title: str, min_chars: int) -> list[dict]:
    """
    Parse a Wikipedia article, return list of ShareGPT samples.
    One sample per meaningful section, or one for the whole intro if no sections.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"}) or soup.find("div", class_="mw-body-content")
    if not content:
        return []

    samples: list[dict] = []

    # Collect intro (everything before first h2)
    intro_parts = []
    for el in content.children:
        if getattr(el, 'name', None) == "h2":
            break
        if getattr(el, 'name', None) in ("p", "ul", "ol", "pre"):
            intro_parts.append(str(el))

    intro_text = clean_text(" ".join(intro_parts))
    if len(intro_text) >= min_chars:
        samples.append({
            "conversations": [
                {"from": "human", "value": f"What is {article_title.replace('_', ' ')}?"},
                {"from": "gpt",   "value": intro_text},
            ],
            "source":   "wikipedia",
            "article":  article_title,
            "section":  "introduction",
        })

    # Collect sections
    current_section = None
    current_parts: list[str] = []

    def flush_section():
        if not current_section or not current_parts:
            return
        if current_section.lower() in SKIP_SECTIONS:
            return
        text = clean_text(" ".join(current_parts))
        if len(text) < min_chars:
            return
        question = f"Explain {current_section} as it relates to {article_title.replace('_', ' ')}."
        samples.append({
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt",   "value": text},
            ],
            "source":   "wikipedia",
            "article":  article_title,
            "section":  current_section,
        })

    for el in content.find_all(["h2", "h3", "p", "ul", "ol", "pre"]):
        if el.name in ("h2", "h3"):
            flush_section()
            current_section = el.get_text().strip()
            current_parts = []
        else:
            if current_section:
                current_parts.append(str(el))

    flush_section()
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia ZIM → GPU corpus JSONL")
    parser.add_argument("zim_path", help="Path to Wikipedia .zim file")
    parser.add_argument("output", help="Output .jsonl path")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Minimum characters per section to include (default: 200)")
    parser.add_argument("--articles", nargs="*",
                        help="Override article list (space-separated titles, use underscores)")
    args = parser.parse_args()

    zim = Archive(args.zim_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    articles = args.articles if args.articles else TARGET_ARTICLES
    total_samples = 0
    found = 0
    not_found = []

    print(f"Wikipedia ZIM: {args.zim_path}")
    print(f"Target articles: {len(articles)}")
    print(f"Output: {out_path}")
    print()

    with open(out_path, "a") as f:
        for title in articles:
            # Try both with and without A/ prefix
            entry = None
            for path in (f"A/{title}", title):
                try:
                    entry = zim.get_entry_by_path(path)
                    break
                except Exception:
                    continue

            if entry is None:
                not_found.append(title)
                print(f"  [not found] {title}")
                continue

            try:
                item = entry.get_item()
                html = bytes(item.content).decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  [error] {title}: {e}")
                continue

            samples = extract_article(html, title, args.min_chars)
            for s in samples:
                f.write(json.dumps(s) + "\n")

            found += 1
            total_samples += len(samples)
            print(f"  [{found}/{len(articles)}] {title.replace('_', ' ')} → {len(samples)} samples")

    print(f"\nDone.")
    print(f"  Articles found:    {found}/{len(articles)}")
    print(f"  Total samples:     {total_samples}")
    if not_found:
        print(f"  Not found ({len(not_found)}): {', '.join(not_found[:10])}")
    print(f"  Output:            {out_path}")


if __name__ == "__main__":
    main()
