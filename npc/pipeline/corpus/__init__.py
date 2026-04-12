"""
Corpus pipeline — pre-processes external sources into local JSONL files
for use as local_file entries in theme.yaml hf_datasets config.

Modules:
    zim_extractor   — Stack Exchange ZIM files (SO, electronics, unix, dsp, etc.)

Planned:
    arxiv           — GPU/compute papers via arXiv API
    github_prs      — Merged ROCm/llama.cpp PRs via GitHub API
    amd_docs        — GPUOpen architecture documentation

All modules output ShareGPT JSONL compatible with dataset.py local_file loading:
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}], ...}
"""
