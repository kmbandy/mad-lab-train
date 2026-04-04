#!/usr/bin/env python3
"""
Generalized synthetic data generator.

Reads all prompts, characters, scenes, and lore config from a theme directory.
Hardware (CUDA/ROCm) is auto-detected.

Usage:
    python3 pipeline/generate.py --config run.yaml --theme themes/dnd_npc --model writer
    python3 pipeline/generate.py --config run.yaml --theme themes/stock_analyst --model analyst
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import yaml
from jinja2 import Template
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Theme loading
# ---------------------------------------------------------------------------

class Theme:
    """Loaded theme — prompts, characters, scenes, config."""

    def __init__(self, theme_dir: Path):
        self.dir = theme_dir
        with open(theme_dir / "theme.yaml") as f:
            self.cfg = yaml.safe_load(f)

        self.name: str = self.cfg["name"]

        # Prompt files
        self._prompt_cache: dict[str, str] = {}

        # Characters
        chars_path = theme_dir / "characters.yaml"
        if chars_path.exists():
            with open(chars_path) as f:
                self.characters: list[dict] = yaml.safe_load(f)["characters"]
        else:
            self.characters = []

        # Scenes
        scenes_path = theme_dir / "scenes.yaml"
        if scenes_path.exists():
            with open(scenes_path) as f:
                self.scenes: dict[str, list[dict]] = yaml.safe_load(f)["scenes"]
        else:
            self.scenes = {}

        # Categories from theme.yaml
        self.categories: list[dict] = self.cfg["categories"]

    def prompt(self, name: str) -> str:
        """Load and cache a prompt file. name = filename without .txt"""
        if name not in self._prompt_cache:
            path = self.dir / "prompts" / f"{name}.txt"
            self._prompt_cache[name] = path.read_text().strip()
        return self._prompt_cache[name]

    def generator_cfg(self, model_key: str) -> dict:
        """Return generator config for a named model key."""
        gens = self.cfg.get("generators", {})
        if model_key not in gens:
            raise ValueError(f"Generator '{model_key}' not defined in theme {self.name}")
        return gens[model_key]

    def reviewer_cfg(self, reviewer_key: str) -> dict:
        """Return reviewer config for a named reviewer key."""
        revs = self.cfg.get("reviewers", {})
        if reviewer_key not in revs:
            raise ValueError(f"Reviewer '{reviewer_key}' not defined in theme {self.name}")
        return revs[reviewer_key]

    def random_scene(self, category: str) -> dict:
        """Pick a random scene for the given category."""
        pool = self.scenes.get(category)
        if not pool:
            # Fall back to first category with scenes
            pool = next(iter(self.scenes.values()), [{}])
        return random.choice(pool)

    def random_character(self) -> dict:
        if not self.characters:
            return {"name": "Unknown NPC", "desc": "A mysterious stranger."}
        return random.choice(self.characters)


# ---------------------------------------------------------------------------
# Lore fetching
# ---------------------------------------------------------------------------

def _fetch_kiwix(url: str, max_chars: int, cache: dict) -> str:
    if url in cache:
        return cache[url]
    try:
        import requests
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", resp.text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        result = text[:max_chars]
        cache[url] = result
        return result
    except Exception as e:
        print(f"  [warn] Kiwix fetch failed ({url}): {e}", file=sys.stderr)
        return ""


def fetch_lore(theme: Theme, run_cfg: dict, query: str) -> tuple[str, str]:
    """
    Fetch lore context. Returns (label, text).
    Tries ChromaDB first, then Kiwix (if configured in theme).
    """
    lore_cfg = theme.cfg.get("lore", {})
    parts: list[str] = []

    # ChromaDB
    chroma_path = run_cfg.get("chromadb_path", "")
    if chroma_path:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma_path)
            col = client.get_collection("memory")
            results = col.query(
                query_texts=[query],
                n_results=3,
                where={"type": {"$in": ["character", "world", "event", "lore"]}},
                include=["documents"],
            )
            docs = (results.get("documents") or [[]])[0]
            if docs:
                parts.append("Relevant lore from campaign records:\n" + "\n---\n".join(docs))
        except Exception as e:
            print(f"  [warn] ChromaDB: {e}", file=sys.stderr)

    # Kiwix
    kiwix_cfg = lore_cfg.get("kiwix", {})
    label = ""
    if kiwix_cfg.get("enabled") and kiwix_cfg.get("topics"):
        kiwix_base = run_cfg.get("kiwix_base", "http://localhost:8091/content/wikipedia_en_all_maxi_2026-02")
        max_chars = kiwix_cfg.get("max_chars", 800)
        topic = random.choice(kiwix_cfg["topics"])
        label = topic
        url = f"{kiwix_base}/{topic.replace(' ', '_')}"
        _cache: dict = {}  # module-level cache would be better in prod
        text = _fetch_kiwix(url, max_chars, _cache)
        if text:
            parts.append(
                f"Reference for inspiration (style/voice guide, do NOT copy):\n{topic}\n{text}"
            )

    lore_text = "\n\n".join(parts) if parts else "(No lore available — use generic context)"
    return label, lore_text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_one(
    client: AsyncOpenAI,
    model_id: str,
    system: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    no_think: bool,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    try:
        async with semaphore:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        content = resp.choices[0].message.content
        if not content:
            return None
        if no_think:
            content = re.sub(r".*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        return content if content else None
    except Exception as e:
        print(f"  [error] generation failed: {e}", file=sys.stderr)
        return None


def validate_output(text: str, theme: Theme) -> bool:
    """
    Basic format validation. Themes can define a 'validation' block in theme.yaml
    with regex patterns that the output must/must_not match.
    """
    val_cfg = theme.cfg.get("validation", {})

    must_match = val_cfg.get("must_match", [])
    for pattern in must_match:
        if not re.search(pattern, text):
            return False

    must_not_match = val_cfg.get("must_not_match", [])
    for pattern in must_not_match:
        if re.search(pattern, text):
            return False

    max_lines = val_cfg.get("max_lines", 0)
    if max_lines:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if len(lines) > max_lines:
            return False

    return True


async def run_generation(
    model_key: str,
    theme: Theme,
    run_cfg: dict,
    target: int,
    out_path: Path,
) -> None:
    gen_cfg = theme.generator_cfg(model_key)
    api_base = run_cfg[f"{model_key}_api_base"]
    model_id  = run_cfg[f"{model_key}_model"]

    system_prompt = theme.prompt(gen_cfg["prompt_file"].replace("prompts/", "").replace(".txt", ""))
    gen_prompt_template = Template(theme.prompt("generation_prompt"))

    temperature = gen_cfg.get("temperature", 0.85)
    max_tokens  = gen_cfg.get("max_tokens", 300)
    no_think    = gen_cfg.get("no_think", False)
    concurrency = run_cfg.get("concurrency", 4)

    cat_names   = [c["name"] for c in theme.categories]
    cat_weights = [c["weight"] for c in theme.categories]

    client    = AsyncOpenAI(base_url=api_base, api_key="unused")
    semaphore = asyncio.Semaphore(concurrency)

    print(f"\n=== Generating {target} samples [{model_key}] → {out_path} ===")

    generated   = 0
    attempts    = 0
    max_attempts = target * 4

    async def attempt_one() -> Optional[dict]:
        category  = random.choices(cat_names, weights=cat_weights, k=1)[0]
        character = theme.random_character()
        scene_entry = theme.random_scene(category)

        lore_label, lore_text = fetch_lore(
            theme, run_cfg, f"{character.get('name', '')} {category}"
        )

        # Render the generation prompt template
        user_prompt = gen_prompt_template.render(
            category=category,
            lore_context=lore_text,
            character=character,
            scene=scene_entry.get("scene", ""),
            mood=scene_entry.get("mood", ""),
            player_action=scene_entry.get("player_action", ""),
            **scene_entry,   # pass all scene fields for custom templates
        )

        response = await generate_one(
            client, model_id, system_prompt, user_prompt,
            temperature, max_tokens, no_think, semaphore,
        )
        if not response or not validate_output(response, theme):
            return None

        return {
            "category":     category,
            "character":    character.get("name", ""),
            "scene":        scene_entry.get("scene", ""),
            "mood":         scene_entry.get("mood", ""),
            "player_action": scene_entry.get("player_action", ""),
            "lore_ref":     lore_label,
            "lore_used":    lore_text != "(No lore available — use generic context)",
            "response":     response,
        }

    with open(out_path, "w") as out_file:
        while generated < target and attempts < max_attempts:
            batch = await asyncio.gather(*[attempt_one() for _ in range(concurrency)])
            for result in batch:
                attempts += 1
                if result is None:
                    print(f"  [{attempts:4d}] skip (bad format)", end="\r")
                    continue
                record = {"id": f"{model_key}_{generated:05d}", "model": model_key, **result}
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                generated += 1
                if generated % 10 == 0 or generated == target:
                    print(f"  [{generated:4d}/{target}] {result['category']} — {result['character']}")
                if generated >= target:
                    break

    print(f"  Done: {generated}/{target} samples ({attempts} attempts)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Run config yaml (model endpoints, paths, etc.)")
    parser.add_argument("--theme",  required=True, help="Path to theme directory")
    parser.add_argument("--model",  required=True, help="Generator key defined in theme.yaml")
    parser.add_argument("--count",  type=int, default=None, help="Override sample target")
    args = parser.parse_args()

    theme_dir = Path(args.theme)
    if not theme_dir.is_absolute():
        theme_dir = Path(__file__).parent.parent / args.theme
    theme = Theme(theme_dir)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent.parent / args.config
    with open(cfg_path) as f:
        run_cfg = yaml.safe_load(f)

    gen_cfg = theme.generator_cfg(args.model)
    target  = args.count or run_cfg.get(f"samples_{args.model}") or run_cfg.get("samples_per_model", 300)

    output_dir = Path(run_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"raw_{args.model}.jsonl"

    asyncio.run(run_generation(args.model, theme, run_cfg, target, out_path))


if __name__ == "__main__":
    main()
