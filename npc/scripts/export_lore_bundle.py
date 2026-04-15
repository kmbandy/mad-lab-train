#!/usr/bin/env python3
"""Export GPU-relevant records from Qdrant to a static JSONL lore bundle for EC2 runs.

Usage:
    python3 scripts/export_lore_bundle.py --output lore_bundle.jsonl
"""

import argparse
import json
import sys

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION  = "memory"
SOURCES     = ["amd_gpu_docs", "github_prs", "arxiv_gpu"]
BATCH_SIZE  = 200


def export(out_path: str) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
    except ImportError:
        print("ERROR: qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    records = []

    for source in SOURCES:
        print(f"  Exporting source: {source}...", end=" ")
        offset = None
        source_count = 0

        scroll_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="source",
                match=qmodels.MatchValue(value=source),
            )]
        )

        while True:
            result, offset = client.scroll(
                collection_name=COLLECTION,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=scroll_filter,
            )
            for point in result:
                payload = point.payload or {}
                content = payload.get("content", "").strip()
                if content:
                    records.append({
                        "source": source,
                        "content": content,
                        "title": payload.get("title", ""),
                        "url": payload.get("url", ""),
                    })
                    source_count += 1

            if offset is None:
                break

        print(f"{source_count} records")

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"  Total: {len(records)} records → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lore_bundle.jsonl")
    args = parser.parse_args()
    export(args.output)
