#!/usr/bin/env python3
"""
Synthetic NPC training data generator.

Pulls lore from ChromaDB and Kiwix, then uses two 27B model passes to generate
NPC dialogue examples in the mad-lab-dnd response format.

Usage:
    python3 generate.py [--config config.yaml] [--model writer|opus|both] [--count N]
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Optional

import chromadb
import requests
import yaml
from openai import AsyncOpenAI

# Number of concurrent generation requests — match llama-server --parallel
CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Kiwix sources — curated character/world articles for lore grounding
# Each entry: (label, kiwix_url)
# The generator picks from these randomly when building prompts so the 27B
# has a rich, focused character reference to draw from.
# ---------------------------------------------------------------------------

KIWIX_BASE = "http://localhost:8091/content/wikipedia_en_all_maxi_2026-02"

KIWIX_SOURCES = [
    # Morally complex, iconic characters
    ("Geralt of Rivia",      f"{KIWIX_BASE}/Geralt_of_Rivia"),
    ("Drizzt Do'Urden",      f"{KIWIX_BASE}/Drizzt_Do%27Urden"),
    ("Artemis Entreri",      f"{KIWIX_BASE}/Drizzt_Do%27Urden"),   # closest available
    ("Strahd von Zarovich",  f"{KIWIX_BASE}/Strahd_von_Zarovich"),
    ("Aragorn",              f"{KIWIX_BASE}/Aragorn"),
    ("Gandalf",              f"{KIWIX_BASE}/Gandalf"),
    ("Boromir",              f"{KIWIX_BASE}/Boromir"),
    ("Frodo Baggins",        f"{KIWIX_BASE}/Frodo_Baggins"),
    ("Catti-brie",           f"{KIWIX_BASE}/Catti-brie"),
    ("Wulfgar",              f"{KIWIX_BASE}/Wulfgar_(Forgotten_Realms)"),
    # World/setting context
    ("Forgotten Realms",     f"{KIWIX_BASE}/Forgotten_Realms"),
    ("Menzoberranzan",       f"{KIWIX_BASE}/Menzoberranzan"),
    ("Underdark",            f"{KIWIX_BASE}/Underdark"),
]

KIWIX_CACHE: dict[str, str] = {}


def fetch_kiwix_article(url: str, max_chars: int = 2000) -> str:
    """Fetch a Kiwix article and return plain text, stripped of HTML."""
    if url in KIWIX_CACHE:
        return KIWIX_CACHE[url]
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        # Strip HTML tags — simple regex is fine for Wikipedia infobox-heavy pages
        text = resp.text
        # Remove script/style blocks
        import re
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        result = text[:max_chars]
        KIWIX_CACHE[url] = result
        return result
    except Exception as e:
        print(f"  [warn] Kiwix fetch failed ({url}): {e}", file=sys.stderr)
        return ""


def pick_kiwix_lore() -> tuple[str, str]:
    """Pick a random Kiwix source and return (label, article_text)."""
    label, url = random.choice(KIWIX_SOURCES)
    text = fetch_kiwix_article(url)
    return label, text


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You are generating synthetic training data for a D&D NPC roleplay model.

Your job: write realistic, in-character NPC responses to player interactions.

Response format — ALWAYS use exactly this structure:
*Physical action or reaction in italics.*
"Spoken dialogue in quotes."

Rules:
- 2-4 sentences maximum — one beat, one reaction, stop
- No narration, no meta-commentary, no stage directions beyond the italics line
- Match the character's voice, vocabulary, and emotional register
- If the character would be guarded, be guarded. If warm, be warm.
- Never use < > or > notation for actions"""

OPUS_SYSTEM = """You are generating synthetic training data for a D&D NPC roleplay model.

Your job: write lore-accurate, character-consistent NPC responses to player interactions.

Response format — ALWAYS use exactly this structure:
*Physical action or reaction in italics.*
"Spoken dialogue in quotes."

Rules:
- 2-4 sentences maximum — one beat, one reaction, stop
- Ground the response in the character's stated history and motivations
- Character actions must be consistent with their relationship axes (affection/respect/familiarity)
- No narration, no meta-commentary
- Never use < > or > notation for actions"""

DRUMMER_SYSTEM = """You are generating synthetic training data for a D&D NPC roleplay model.

Your job: write vivid, immersive NPC responses that feel alive — specific to this character, not generic fantasy.

Response format — ALWAYS use exactly this structure:
*Physical action or reaction in italics.*
"Spoken dialogue in quotes."

Rules:
- 2-4 sentences maximum — one beat, one reaction, stop
- Action line should be physical and specific — what the body does, not what the face shows
- Dialogue should sound like speech — contractions, rhythm, the way this person actually talks
- Every response should be unmistakably this character, not interchangeable with any other NPC
- Never use < > or > notation for actions"""

GENERATION_PROMPT = """Generate one training example for category: {category}

{lore_context}

Character: {character_name}
{character_desc}

Scene: {scene}
Mood: {mood}
Player action: {player_action}

Write the NPC's response now (remember: *italics action* then "dialogue", 2-4 sentences max):"""

# ---------------------------------------------------------------------------
# Scene pools
# ---------------------------------------------------------------------------

FALLBACK_CHARACTERS = [
    # Original cast
    {
        "name": "Mira Valdessen",
        "desc": "A cold, calculating merchant guildmaster. Affection: 3, Respect: 7, Familiarity: 4. Values leverage and information above all.",
    },
    {
        "name": "Dorian Ashveil",
        "desc": "A fallen archmage, bitter but proud. Affection: 5, Respect: 8, Familiarity: 2. Hides vulnerability behind sharp wit.",
    },
    {
        "name": "The Tavern Keeper",
        "desc": "A weathered innkeeper who has seen everything. Affection: 6, Respect: 5, Familiarity: 3. Pragmatic and quietly observant.",
    },
    {
        "name": "Captain Reldris",
        "desc": "A city guard captain, loyal to the crown but fair. Affection: 4, Respect: 6, Familiarity: 3. Follows the law, dislikes complications.",
    },
    {
        "name": "The Fence",
        "desc": "A nervous black market dealer, always watching the door. Affection: 3, Respect: 3, Familiarity: 5. Survival-focused, easily spooked.",
    },
    {
        "name": "Sister Vayne",
        "desc": "A temple healer with a dark secret. Affection: 7, Respect: 6, Familiarity: 2. Compassionate on the surface, deeply conflicted beneath.",
    },
    {
        "name": "Lord Harwick",
        "desc": "A minor noble with ambitions far beyond his station. Affection: 2, Respect: 4, Familiarity: 1. Condescending to those he deems beneath him.",
    },
    # Expanded cast
    {
        "name": "Zelara the Undying",
        "desc": "An ancient lich who grew bored of conquest. Affection: 2, Respect: 9, Familiarity: 1. Eerily polite, treats mortals like amusing insects.",
    },
    {
        "name": "Bram Coldwater",
        "desc": "A retired assassin turned fisherman. Affection: 5, Respect: 4, Familiarity: 6. Warm on the surface but always watching exits.",
    },
    {
        "name": "The Clockwork Oracle",
        "desc": "A construct that speaks only in riddles and half-truths. Affection: 0, Respect: 6, Familiarity: 2. Neither kind nor cruel — simply inevitable.",
    },
    {
        "name": "Thessaly Brynn",
        "desc": "A halfling scout with a sharp tongue and sharper ears. Affection: 7, Respect: 5, Familiarity: 8. Irreverent, loyal to those who earn it.",
    },
    {
        "name": "Elder Mosswick",
        "desc": "A druidic elder who speaks for the forest. Affection: 4, Respect: 7, Familiarity: 2. Slow to anger but devastating when roused.",
    },
    {
        "name": "Commander Serath",
        "desc": "A battle-hardened drow general in exile. Affection: 3, Respect: 8, Familiarity: 3. Disciplined, proud, carries deep shame she never shows.",
    },
    {
        "name": "Pip the Urchin",
        "desc": "A street child who knows every secret in the city. Affection: 6, Respect: 2, Familiarity: 7. Cheerful and relentless, survival instinct honed razor-sharp.",
    },
    {
        "name": "The Loremaster",
        "desc": "A blind scholar who has read every text in the great library. Affection: 5, Respect: 9, Familiarity: 4. Gentle but ruthlessly precise with words.",
    },
    {
        "name": "Vareth Duskmantle",
        "desc": "A warlock whose patron grows increasingly demanding. Affection: 4, Respect: 5, Familiarity: 5. Anxious beneath a veneer of confidence, cracking at the edges.",
    },
    {
        "name": "Mother Grell",
        "desc": "A crime boss who runs the city's thieves guild as a matriarch. Affection: 6, Respect: 9, Familiarity: 3. Generous to family, merciless to enemies.",
    },
    {
        "name": "Ser Aldric the Broken",
        "desc": "A once-legendary paladin who lost his faith after a massacre he caused. Affection: 4, Respect: 6, Familiarity: 2. Haunted, seeking redemption he believes he doesn't deserve.",
    },
    {
        "name": "Nyssa Dawnwhisper",
        "desc": "A bard whose songs have toppled two kings. Affection: 8, Respect: 7, Familiarity: 6. Charming and dangerous — always performing, even when alone.",
    },
]

SCENES_BY_CATEGORY = {
    "npc_dialogue": [
        ("The Red Lantern Tavern, evening. Crowded but not rowdy.", "Neutral", "The player sits down across from the NPC and asks about recent events in town."),
        ("A back-alley meeting, night. Rain pattering on cobblestones.", "Cautious", "The player slides a coin purse across the table without a word."),
        ("Guild hall antechamber. Formal and uncomfortable.", "Guarded", "The player presents a letter of introduction from a mutual contact."),
        ("Marketplace stall, midday. Busy and loud.", "Distracted", "The player asks about a specific item they need urgently."),
        ("Temple entrance, dawn. Quiet and reverent.", "Composed", "The player asks for information about a missing person."),
        ("A private study, firelight flickering.", "Thoughtful", "The player asks the NPC for their opinion on a difficult decision."),
        ("A ship's cabin, mid-voyage. The sea is rough.", "Tense", "The player asks the NPC why they really took this job."),
        ("Underground fighting pit, between bouts.", "Wired", "The player approaches the NPC with a proposition."),
        ("A moonlit garden. The party is audible inside.", "Conspiratorial", "The player pulls the NPC aside away from the crowd."),
        ("Roadside camp at dusk. Shared fire, strangers.", "Wary", "The player offers the NPC food and asks where they're headed."),
        ("A wizard's tower, waiting room. Strange sounds above.", "Anxious", "The player strikes up conversation while they both wait."),
        ("Prison visiting room. Guards nearby.", "Controlled", "The player has come to deliver a message — and possibly an offer."),
    ],
    "npc_confrontation": [
        ("Dungeon corridor. Torchlight flickers. No escape route visible.", "Hostile", "The player has been caught somewhere they shouldn't be."),
        ("Nobleman's study. Guards outside the door.", "Cold fury", "The player has just accused the NPC of betrayal to their face."),
        ("City gate checkpoint. Soldiers present.", "Suspicious", "The player's papers don't quite add up and the NPC has noticed."),
        ("Ruined tower, exposed to the elements.", "Desperate", "The player holds something the NPC desperately needs."),
        ("Banquet hall. Other guests are watching.", "Controlled rage", "The player has just publicly contradicted the NPC's version of events."),
        ("A dark forest road, torches extinguished.", "Predatory", "The NPC has been waiting for the player. This was not a coincidence."),
        ("Throne room. The court watches in silence.", "Imperial", "The player has spoken out of turn and the NPC will not let it pass."),
        ("Burning building. Both are trapped.", "Frantic", "The player demands answers the NPC doesn't want to give."),
        ("Arena stands. The crowd is restless.", "Contemptuous", "The player has challenged the NPC's authority publicly."),
        ("Narrow bridge over a chasm. No room to pass.", "Calculating", "The NPC blocks the way and names their price."),
    ],
    "npc_revelation": [
        ("Private room at an inn. Candles burning low.", "Reluctant", "The player has finally earned enough trust. The NPC decides to tell the truth."),
        ("Abandoned safehouse. Dust on everything.", "Haunted", "The player asks about the NPC's past — the question they've been avoiding."),
        ("Rooftop overlooking the city, late at night.", "Resigned", "The player has pieced it together and confronts the NPC gently."),
        ("Dying campfire. Everyone else is asleep.", "Vulnerable", "The player sits in silence and waits. The NPC finally speaks."),
        ("Church confessional. Anonymity implied.", "Ashamed", "The player asks what the NPC has done that they regret most."),
        ("Graveyard at dusk. Fresh flowers on a headstone.", "Grief-stricken", "The player follows the NPC here and waits to be acknowledged."),
        ("A locked vault. The NPC holds the only key.", "Conflicted", "The player has figured out what's inside and asks why it's hidden."),
        ("Rain-soaked battlements. The city below is quiet.", "Exhausted", "The player asks the NPC if they ever regret the choices that led here."),
    ],
    "npc_ambient": [
        ("Busy market street. The NPC is working.", "Busy", "The player asks for simple directions."),
        ("Inn common room. The NPC is eating alone.", "Tired", "The player sits nearby and makes small talk."),
        ("Stable yard, morning. Horses being readied.", "Content", "The player asks if the NPC has heard anything interesting lately."),
        ("Library. The NPC is reading.", "Absorbed", "The player interrupts to ask a simple question."),
        ("Temple courtyard. The NPC is tending the garden.", "Peaceful", "The player asks about the local history of the area."),
        ("Dockside, early morning. Gulls overhead.", "Indifferent", "The player tries to make conversation with a stranger."),
        ("Smithy, mid-afternoon. The forge is hot.", "Focused", "The player watches the NPC work and asks about the craft."),
        ("Herbalist's shop. Bundles of dried plants everywhere.", "Methodical", "The player asks if the NPC has anything for a specific ailment."),
        ("Festival crowd, midday. Music and laughter nearby.", "Festive", "The player spots the NPC and waves them over."),
        ("Gatehouse, night watch. Stars visible above.", "Contemplative", "The player relieves the NPC's boredom with a question about the stars."),
        ("Apothecary back room. Shelves of unlabeled vials.", "Secretive", "The player asks about something that isn't on the public shelves."),
    ],
}

# ---------------------------------------------------------------------------
# ChromaDB lore fetching
# ---------------------------------------------------------------------------

def fetch_chromadb_lore(chromadb_path: str, query: str, n: int = 3) -> str:
    """Pull relevant lore from ChromaDB. Returns formatted string or empty."""
    try:
        client = chromadb.PersistentClient(path=chromadb_path)
        col = client.get_collection("memory")
        results = col.query(
            query_texts=[query],
            n_results=n,
            where={"type": {"$in": ["character", "world", "event", "lore"]}},
            include=["documents"],
        )
        docs_list = results.get("documents") or [[]]
        docs = docs_list[0] if docs_list else []
        if not docs:
            return ""
        return "Relevant lore from campaign records:\n" + "\n---\n".join(docs)
    except Exception as e:
        print(f"  [warn] ChromaDB fetch failed: {e}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(
    category: str,
    character: dict,
    chromadb_lore: str,
    kiwix_label: str,
    kiwix_text: str,
) -> tuple[str, str, str, str]:
    """Return (prompt, scene, mood, player_action)."""
    scenes = SCENES_BY_CATEGORY.get(category, SCENES_BY_CATEGORY["npc_dialogue"])
    scene, mood, player_action = random.choice(scenes)

    lore_parts = []
    if chromadb_lore:
        lore_parts.append(chromadb_lore)
    if kiwix_text:
        lore_parts.append(
            f"Reference character for inspiration (do NOT copy — use as a style/voice guide): "
            f"{kiwix_label}\n{kiwix_text}"
        )
    lore_block = "\n\n".join(lore_parts) if lore_parts else "(No lore available — use generic fantasy context)"

    prompt = GENERATION_PROMPT.format(
        category=category,
        lore_context=lore_block,
        character_name=character["name"],
        character_desc=character["desc"],
        scene=scene,
        mood=mood,
        player_action=player_action,
    )
    return prompt, scene, mood, player_action


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

async def generate_sample(
    client: AsyncOpenAI,
    model: str,
    system: str,
    prompt: str,
    temperature: float = 0.85,
    no_think: bool = False,
) -> Optional[str]:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800 if no_think else 300,
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        if not content:
            return None
        # Strip everything up to and including </think> (handles mixed-case and preamble)
        import re
        content = re.sub(r".*?</think>", "", content, flags=re.DOTALL|re.IGNORECASE).strip()
        return content if content else None
    except Exception as e:
        print(f"  [error] generation failed: {e}", file=sys.stderr)
        return None


def is_valid_format(text: str) -> bool:
    """Basic format check — must have italics action and quoted dialogue."""
    has_italics = text.count("*") >= 2
    has_quotes = '"' in text
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    too_long = len(lines) > 8
    return has_italics and has_quotes and not too_long


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_generation(
    model_name: str,
    api_base: str,
    model_id: str,
    system_prompt: str,
    target: int,
    categories: list,
    chromadb_path: str,
    out_path: Path,
) -> None:
    print(f"\n=== Generating {target} samples [{model_name}] → {out_path} ===")

    client = AsyncOpenAI(base_url=api_base, api_key="unused")
    cat_names = [c["name"] for c in categories]
    cat_weights = [c["weight"] for c in categories]
    semaphore = asyncio.Semaphore(CONCURRENCY)

    generated = 0
    attempts = 0
    max_attempts = target * 4

    async def attempt_one() -> Optional[dict]:
        category = random.choices(cat_names, weights=cat_weights, k=1)[0]
        character = random.choice(FALLBACK_CHARACTERS)
        chromadb_lore = fetch_chromadb_lore(chromadb_path, f"{character['name']} {category}")
        kiwix_label, kiwix_text = pick_kiwix_lore()
        prompt, scene, mood, player_action = build_prompt(
            category, character, chromadb_lore, kiwix_label, kiwix_text
        )
        async with semaphore:
            response = await generate_sample(client, model_id, system_prompt, prompt,
                                             no_think=(model_name in ("opus",)))
        if not response or not is_valid_format(response):
            return None
        return {
            "category": category,
            "character": character["name"],
            "scene": scene,
            "mood": mood,
            "player_action": player_action,
            "kiwix_ref": kiwix_label,
            "lore_used": bool(chromadb_lore),
            "response": response,
        }

    with open(out_path, "w") as out_file:
        while generated < target and attempts < max_attempts:
            # Fire CONCURRENCY requests at once
            batch = await asyncio.gather(*[attempt_one() for _ in range(CONCURRENCY)])
            for result in batch:
                attempts += 1
                if result is None:
                    print(f"  [{attempts:4d}] skip (bad format)", end="\r")
                    continue
                record = {"id": f"{model_name}_{generated:05d}", "model": model_name, **result}
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                generated += 1
                if generated % 10 == 0 or generated == target:
                    print(f"  [{generated:4d}/{target}] {result['category']} — {result['character']} (ref: {result['kiwix_ref']})")
                if generated >= target:
                    break

    print(f"  Done: {generated} samples written ({attempts} attempts)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["writer", "opus", "drummer", "both"], default="both")
    parser.add_argument("--count", type=int, default=None, help="Override samples_per_model")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = cfg["categories"]
    chromadb_path = cfg["chromadb_path"]

    models_to_run = []
    if args.model in ("writer", "both"):
        models_to_run.append(("writer", cfg["writer_api_base"], cfg["writer_model"], WRITER_SYSTEM))
    if args.model in ("opus", "both"):
        models_to_run.append(("opus", cfg["opus_distill_api_base"], cfg["opus_distill_model"], OPUS_SYSTEM))
    if args.model == "drummer":
        models_to_run.append(("drummer", cfg["drummer_api_base"], cfg["drummer_model"], DRUMMER_SYSTEM))

    for model_name, api_base, model_id, system_prompt in models_to_run:
        target = args.count or cfg.get(f"samples_{model_name}") or cfg["samples_per_model"]
        out_path = output_dir / f"raw_{model_name}.jsonl"
        asyncio.run(run_generation(
            model_name, api_base, model_id, system_prompt,
            target, categories, chromadb_path, out_path,
        ))


if __name__ == "__main__":
    main()
