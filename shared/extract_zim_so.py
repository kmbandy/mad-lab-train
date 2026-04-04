#!/usr/bin/env python3
"""
Extract Q&A pairs from a Stack Overflow (or Stack Exchange) ZIM file.

Outputs JSONL with accepted-answer pairs only, suitable for fine-tuning.

Usage:
    python3 extract_zim_so.py <zim_path> <output_jsonl> [--max N] [--min-score N]

Output format (ShareGPT-compatible):
    {"conversations": [
        {"from": "human", "value": "<question title>\n\n<question body>"},
        {"from": "gpt",   "value": "<accepted answer body>"}
    ], "source": "stackoverflow", "score": 42, "question_id": "12345"}
"""

import argparse
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from libzim.reader import Archive


def clean_html(html_fragment: str) -> str:
    """Strip HTML tags, normalize whitespace."""
    soup = BeautifulSoup(html_fragment, "html.parser")
    # Preserve code blocks with markers
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text()}`")
    for pre in soup.find_all("pre"):
        pre.replace_with(f"\n```\n{pre.get_text().strip()}\n```\n")
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
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


def extract_entry(html: str) -> dict | None:
    """
    Parse a SO question page and return a Q&A pair if it has an accepted answer.
    Returns None if no accepted answer or parsing fails.
    """
    soup = BeautifulSoup(html, "html.parser")

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

    # Score on question
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

    return {
        "conversations": [
            {"from": "human", "value": question_text},
            {"from": "gpt",   "value": answer_text},
        ],
        "source": "stackoverflow",
        "q_score": q_score,
        "ans_score": ans_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract SO ZIM → fine-tune JSONL")
    parser.add_argument("zim_path", help="Path to .zim file")
    parser.add_argument("output", help="Output .jsonl path")
    parser.add_argument("--max", type=int, default=0, help="Max entries to extract (0=all)")
    parser.add_argument("--min-score", type=int, default=1, help="Min question score to include (default: 1)")
    parser.add_argument("--min-ans-score", type=int, default=0, help="Min accepted answer score (default: 0)")
    args = parser.parse_args()

    zim = Archive(args.zim_path)
    total = zim.entry_count
    print(f"ZIM: {args.zim_path}")
    print(f"Total entries: {total:,}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped_no_accepted = 0
    skipped_low_score = 0
    skipped_parse = 0
    examined = 0

    with open(out_path, "w") as f:
        for i in range(total):
            entry = zim._get_entry_by_id(i)

            # Only process question pages (a/ namespace)
            if not entry.path.startswith("a/"):
                continue

            examined += 1

            try:
                item = entry.get_item()
                html = bytes(item.content).decode("utf-8", errors="replace")
            except Exception:
                skipped_parse += 1
                continue

            result = extract_entry(html)

            if result is None:
                skipped_no_accepted += 1
                continue

            if result["q_score"] < args.min_score:
                skipped_low_score += 1
                continue

            if result["ans_score"] < args.min_ans_score:
                skipped_low_score += 1
                continue

            f.write(json.dumps(result) + "\n")
            extracted += 1

            if extracted % 1000 == 0:
                pct = i / total * 100
                print(f"  [{pct:.1f}%] extracted={extracted:,}  no_accepted={skipped_no_accepted:,}  low_score={skipped_low_score:,}", flush=True)

            if args.max and extracted >= args.max:
                break

    print(f"\nDone.")
    print(f"  Examined:          {examined:,}")
    print(f"  Extracted:         {extracted:,}")
    print(f"  No accepted ans:   {skipped_no_accepted:,}")
    print(f"  Low score:         {skipped_low_score:,}")
    print(f"  Parse errors:      {skipped_parse:,}")
    print(f"  Output:            {out_path}")


if __name__ == "__main__":
    main()
