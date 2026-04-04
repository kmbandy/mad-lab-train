# Mad-Lab Fine-Tune Pipeline — Reference Guide

A generalized synthetic data generation + QLoRA fine-tuning pipeline.
Works for any domain by swapping a **theme directory** and a **run config**.

---

## Why this design?

The goal is to produce training data with genuine variety and nuance — not just a flat
list of `{question, answer}` pairs. By separating:

- **Who** (character) — the subject of every sample
- **What kind** (category) — the type of knowledge being exercised
- **Context** (scene/mood) — the situation the user is in when asking
- **The ask** (player_action) — the actual question

...the same underlying topic (e.g. Charlemagne, or a tomato plant, or $NVDA) generates
dozens of meaningfully different training samples that teach the model to adapt its
register, depth, and style to context — not just recall facts.

Same character + different scene = the model learns to answer a frantic student
differently than a researcher writing a paper.

---

## File structure

```
training/
  pipeline/
    run.py          # orchestrator — runs stages end-to-end
    generate.py     # synthetic sample generation (async, all generators)
    validate.py     # fast-filter + cross-review quality passes
    dataset.py      # merge validated + HF data → train/eval split
    train.py        # QLoRA fine-tuner (hardware-aware)
    hardware.py     # CUDA/ROCm auto-detection

  themes/
    <theme_name>/
      theme.yaml          # domain config — drives everything
      characters.yaml     # the "who" — sampled per generation
      scenes.yaml         # the "context" — grouped by category
      finetune.yaml       # training hyperparams (model, LoRA, epochs)
      prompts/
        generator_<key>.txt     # LLM system prompt for each generator
        generator_<key2>.txt
        generation_prompt.txt   # Jinja2 template — builds the user turn sent to LLM
        fast_filter_system.txt  # fast filter judge system prompt
        fast_filter_prompt.txt  # fast filter judge user prompt (Jinja2)
        reviewer_<key>.txt      # cross-reviewer system prompt
        cross_review_prompt.txt # cross-reviewer user prompt (Jinja2)
        system_prompt.txt       # system turn injected into every training sample
        human_turn.txt          # Jinja2 — user turn in every training sample

  run_<theme>.yaml      # infrastructure config (API endpoints, targets, paths)
```

---

## The two config files

Everything is driven by two configs passed to every pipeline command:

### `run_<theme>.yaml` — infrastructure (the *what* and *how many*)

```yaml
# API endpoints — one entry per generator/reviewer key defined in theme.yaml
writer_api_base: "http://192.168.1.15:8080/v1"
writer_model: "default"
qwen_api_base: "http://192.168.1.15:8080/v1"
qwen_model: "default"

# Fast filter always runs on local GPU (stop other llama-server instances first)
fast_filter_api_base: "http://127.0.0.1:8080/v1"
fast_filter_model: "omnicoder"

# Lore sources
chromadb_path: "/home/kmbandy/.mad-lab-mcp/chromadb"
kiwix_base: "http://localhost:8091/content/wikipedia_en_all_maxi_2026-02"

# Sample targets — key pattern is samples_<generator_key>
samples_writer:  300
samples_opus:    150
samples_per_model: 300   # fallback if specific key not set

concurrency: 4            # parallel async requests to llama-server

# Validation thresholds
min_quality_score: 0.6        # fast filter pass line
cross_review_threshold: 0.80  # reviewer pass line

# Dataset
hf_target_total: 2500
output_dir: "/home/kmbandy/mad-lab-dnd/training/data/pipeline_out"
```

**Pattern for API keys:** `{generator_key}_api_base` and `{generator_key}_model`.
If your theme defines a generator named `analyst`, the run config needs
`analyst_api_base` and `analyst_model`. Same for reviewers.

### `theme.yaml` — domain knowledge (the *what kind*)

Defines the structure of the domain — generators, reviewers, categories, validation
rules, lore sources, dataset format, HF mixing. See full example below.

---

## Pipeline stages

```
generate → validate → dataset → train
```

Run all stages:
```bash
python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc
```

Run specific stages (e.g. resume after generation is done):
```bash
python3 pipeline/run.py --config run_dnd_npc.yaml --theme themes/dnd_npc \
    --stages dataset train
```

Other flags:
```
--no-hf          skip HuggingFace dataset mixing
--no-regen       skip generate if raw output files already exist
--dry-run        print commands without executing
--keep-going     continue past stage failures
--eval-split N   fraction for eval set (default 0.1)
```

### Stage 1 — generate

Calls `generate.py` once per generator defined in `theme.yaml`.

For each call, the generator:
1. Picks a random **category** (weighted by theme config)
2. Picks a random **character** from `characters.yaml`
3. Picks a random **scene** for that category from `scenes.yaml`
4. Fetches **lore** from ChromaDB + Kiwix (if configured)
5. Renders `prompts/generation_prompt.txt` (Jinja2) with all of the above
6. Sends to LLM with `prompts/generator_<key>.txt` as the system prompt
7. Validates output against `theme.yaml` `validation` block
8. Writes passing samples to `data/raw_<key>.jsonl`

### Stage 2 — validate

Two sub-passes per generator:

**Fast filter** — local model scores each raw sample (0.0–1.0).
Uses `prompts/fast_filter_system.txt` + `prompts/fast_filter_prompt.txt`.
Threshold: `min_quality_score` from run config.
Output: `data/filtered_<key>.jsonl`

**Cross-review** — a different model reviews filtered samples.
Uses `prompts/reviewer_<reviewer_key>.txt` + `prompts/cross_review_prompt.txt`.
Threshold: `cross_review_threshold` from run config.
Output: `data/validated_<key>.jsonl`

Score fields are embedded in each sample record for later analysis.

### Stage 3 — dataset

Reads all `validated_<key>.jsonl` files + optional HF datasets.
Converts each sample into ShareGPT conversation format:
```json
{
  "conversations": [
    {"from": "system", "value": "<system_prompt.txt content>"},
    {"from": "human",  "value": "<human_turn.txt rendered with sample fields>"},
    {"from": "gpt",    "value": "<sample.response>"}
  ],
  "_meta": { "source": "synthetic", "fast_filter_score": 0.82, ... }
}
```
Outputs: `dataset.jsonl`, `train.jsonl`, `eval.jsonl`

### Stage 4 — train

QLoRA fine-tune via trl SFTTrainer.
Hardware auto-detected — fp16 on GTX 1070 (sm_61), bf16 on Ampere+, ROCm supported.
Hyperparams from `finetune.yaml` in the theme directory.
Output: LoRA adapter in `finetune.yaml:output_dir`.

---

## Creating a new theme — decision framework

### 1. What is a "character"?

The subject of every generated sample. Randomly sampled per generation.
Ask: *what is this model being asked about?*

| Domain | Character |
|--------|-----------|
| D&D NPC roleplay | NPC with name, race, personality |
| Stock analyst | Ticker with sector, market cap, notes |
| Gardening | Plant with variety, season, difficulty |
| Medieval history | Historical figure/event/concept with period, dates |
| Cooking | Dish or ingredient with cuisine, technique, difficulty |

Characters live in `characters.yaml`. Fields can be anything — they're all passed
to the generation prompt template as `{{ character.field_name }}`.

### 2. What are "scenes"?

The specific situation the user is in when asking. Grouped by category.
Ask: *what context shapes how this question should be answered?*

Two valid approaches:

**In-world context** (D&D style) — the scene is part of the domain itself:
```yaml
- scene: "dimly lit tavern, midnight"
  mood: "suspicious"
  player_action: "A stranger slides a coin across the table and asks for information"
```

**Real-world context** (educational assistant style) — the scene is the learner's situation:
```yaml
- scene: "college dorm room, night before an exam"
  mood: "frantic"
  player_action: "Give me the most important things I need to know"
```

The second approach is powerful for educational/assistant models because it teaches
the model to adapt its register — same topic, cramming student vs. paper-writer
vs. documentary-curious = three completely different appropriate responses.

Scenes live in `scenes.yaml` under their category key. Any fields you add to a
scene entry are automatically passed to the generation prompt template.

### 3. What are categories?

The types of knowledge or interaction the model needs to handle.
Weights determine how often each is sampled — match the real distribution you want.

Common patterns:
- **Time periods** (history): Early/High/Late Middle Ages
- **Topic types** (knowledge domain): earnings analysis, technical patterns, risk assessment
- **Interaction types** (roleplay): confrontation, revelation, ambient, dialogue
- **Task types** (assistant): diagnosis, planning, how-to, troubleshooting

You can mix approaches. Thematic categories (causes_and_effects, myth_vs_reality)
can cut across time periods.

### 4. What does validation enforce?

Use `must_match` only when there's a hard format requirement in every response.
Use `must_not_match` to block failure modes (AI refusals, hedging, wrong format).
Use `max_lines` to prevent runaway long-form outputs.

| Domain | Validation |
|--------|------------|
| D&D NPC | must have `*italics*` and `"quotes"`, max 8 lines |
| Stock analyst | must have `$TICKER` symbol |
| Gardening | must_not_match refusal phrases, max 15 lines |
| Medieval history | must_not_match refusals/hedging, optionally require a year |

### 5. How many generators and reviewers?

- **1 generator, 1 reviewer** — simplest, for domains where style variety doesn't matter much
- **2 generators, 1 reviewer** — primary (practical) + secondary (technical/deeper)
- **3 generators, 3 reviewers** (D&D) — maximum diversity, also maximum API cost

Cross-review pairs each generator with each reviewer — 3×3 = 9 review passes.
Start simple and add generators if the output quality plateaus.

### 6. What lore sources?

- **Kiwix** — Wikipedia offline. Pick topics relevant to your domain. The generator gets
  a random excerpt each call to ground the response. Works out of the box if Kiwix is running.
- **ChromaDB** — campaign records, custom docs, scraped content. Queried with the
  character + category as the search string.
- **static_file** — a single text file always injected. Good for strategy rules,
  world context, style guides.

---

## Worked examples

### D&D NPC Roleplay

```
character = NPC (Elara Dawnwhisper, High Elf Ranger, secretive, poetic)
category  = npc_dialogue
scene     = dimly lit tavern, midnight, single candle
mood      = guarded
player_action = "The player slides a coin toward her and asks about the road north"

→ Response: in-character dialogue with *action beats* and "spoken words"
```

Validation enforces `*italics*` and `"quotes"` because that's a hard format requirement.
The model learns to stay in character AND use the right format.

### Stock Analyst

```
character = $NVDA (Semiconductors, ~$2T market cap, AI/GPU leader)
category  = technical_pattern
scene     = after-hours following a 5% gap up on earnings beat
mood      = momentum
player_action = "Is this breakout sustainable or a trap?"

→ Response: cites RSI, volume, price level, states directional bias, names an invalidation level
```

Validation requires `$TICKER` in every response — the model learns it must always reference data.

### Gardening Assistant

```
character = Tomato (Cherokee Purple heirloom, indeterminate, heavy feeder)
category  = pest_disease
scene     = backyard garden, noticed the problem this morning
mood      = alarmed
player_action = "White powdery coating is appearing on my leaves, is it treatable?"

→ Response: diagnoses powdery mildew, names organic treatment (neem oil/baking soda),
  explains what worsens it (poor airflow), gives a first step
```

Scene is the real-world gardener context. Model learns to give actionable first steps,
not textbook definitions.

### Medieval History Assistant

```
character = Charlemagne (Early Middle Ages, 742–814 CE, Frankish Emperor)
category  = early_middle_ages
scene     = college dorm room, night before an exam
mood      = frantic
player_action = "Give me the most important things I need to know for my test"

→ Response: punchy bullet points — key dates, why he matters, Carolingian Renaissance
  in 3-4 sentences. No lengthy historiography.
```

Same character, different scene:
```
scene     = writing a 10-page research paper
mood      = focused
player_action = "What do historians debate about his legacy?"

→ Response: multiple scholarly perspectives, acknowledges complexity, more nuanced
```

The model learns to read context and modulate depth/style accordingly.

---

## The prompts in detail

### `generation_prompt.txt` — what the LLM sees when generating

This is the Jinja2 template rendered per-sample. It receives:
- `{{ category }}` — from theme categories
- `{{ character }}` — the full character dict (access fields as `{{ character.name }}`)
- `{{ scene }}` / `{{ mood }}` / `{{ player_action }}` — from scenes.yaml
- `{{ lore_context }}` — fetched from ChromaDB/Kiwix
- Any additional scene fields via `**scene_entry`

Keep it structured. Tell the model exactly what to output and what not to include.
"Output only the response text, no labels or preamble" is critical.

### `generator_<key>.txt` — system prompt for generation

Who the LLM is when generating samples. Be specific about:
- Voice and style
- Format constraints (length, structure)
- What to always include
- What to never do

Different generators should have meaningfully different system prompts, not just
cosmetically different ones. A "critic" generator should genuinely generate
contrarian/bearish takes, not just slightly different phrasing.

### `system_prompt.txt` — injected into every training sample

This becomes the `system` turn in every conversation in the dataset.
It's what the fine-tuned model "believes" it is.
Keep it concise and authoritative. This is the model's identity post-training.

### `human_turn.txt` — user turn in every training sample

Jinja2 template. Uses the sample's fields. Keep it natural — this is what users
will actually type to interact with the trained model.

```
# D&D style (in-world context)
[CHARACTER: {{ character }}] [SCENE: {{ scene }}] [MOOD: {{ mood }}] [PLAYER ACTION: {{ player_action }}]

# Educational assistant style (real-world context)
[TOPIC: {{ character }}] [CONTEXT: {{ scene }}] {{ player_action }}

# Or even more natural
{{ player_action }}
```

More structured human turns = more predictable model behavior at inference time.
Less structured = more natural but less controllable.

### `fast_filter_system.txt` + `fast_filter_prompt.txt`

The fast filter is a cheap local model scoring 0.0–1.0. It needs to be:
- Specific about what earns high vs low scores
- Asked to output ONLY a decimal number (max_tokens: 10)

The prompt template renders with all sample fields. Include the character, scene,
and response so the judge can evaluate fit, not just output quality in isolation.

### `reviewer_<key>.txt` + `cross_review_prompt.txt`

Cross-review uses a different model than the generator. It scores 0.0–1.0 with
a one-sentence reason. The system prompt defines the reviewer's judgment criteria.
Each reviewer key needs a corresponding `{key}_api_base` and `{key}_model` in the run config.

---

## `finetune.yaml` — training hyperparams

Most fields are hardware-dependent, not domain-dependent.
For GTX 1070 (8GB VRAM, sm_61, fp16 only):

```yaml
base_model: /home/kmbandy/models/SmolLM3-3B-Base   # change per fine-tune
output_dir: /path/to/output/adapter                  # change per fine-tune

# These are GTX 1070 constraints — don't change unless hardware changes
num_epochs: 3
micro_batch_size: 1
gradient_accumulation_steps: 16   # effective batch = 16
learning_rate: 2e-4
warmup_steps: 20
sequence_len: 2048
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

Only `base_model` and `output_dir` change between themes on the same hardware.

---

## Checklist for a new theme

```
[ ] theme.yaml
    [ ] generators (at least 1 primary)
    [ ] reviewers (at least 1)
    [ ] categories with weights summing to ~1.0
    [ ] validation block (must_not_match at minimum)
    [ ] fast_filter block
    [ ] cross_review block
    [ ] dataset block (system_prompt, human_turn, response_field, generator_keys)
    [ ] output block (system_role, user_role, assistant_role, chat_template)
    [ ] lore block (kiwix topics relevant to domain)
    [ ] hf_datasets ([] is fine to start)

[ ] characters.yaml
    [ ] 10-20 entries minimum for variety
    [ ] Fields that are meaningful for generation_prompt.txt template

[ ] scenes.yaml
    [ ] At least 3-4 scenes per category
    [ ] scene / mood / player_action fields (or custom fields)
    [ ] Scenes are specific enough that the LLM can write a specific response

[ ] finetune.yaml
    [ ] base_model path
    [ ] output_dir path

[ ] prompts/
    [ ] generator_<key>.txt for each generator
    [ ] generation_prompt.txt (Jinja2)
    [ ] fast_filter_system.txt
    [ ] fast_filter_prompt.txt (Jinja2 with {{ response }} and context fields)
    [ ] reviewer_<key>.txt for each reviewer
    [ ] cross_review_prompt.txt (Jinja2)
    [ ] system_prompt.txt (model identity for training)
    [ ] human_turn.txt (Jinja2 — user turn template)

[ ] run_<theme>.yaml
    [ ] {generator_key}_api_base and {generator_key}_model for each generator
    [ ] {reviewer_key}_api_base and {reviewer_key}_model for each reviewer
    [ ] fast_filter_api_base and fast_filter_model
    [ ] samples_{key} for each generator
    [ ] output_dir
    [ ] min_quality_score and cross_review_threshold
```

---

## Quick reference — run commands

```bash
# Full pipeline
python3 pipeline/run.py --config run_X.yaml --theme themes/X

# Generation only (all generators in theme)
python3 pipeline/run.py --config run_X.yaml --theme themes/X --stages generate

# Validate only
python3 pipeline/run.py --config run_X.yaml --theme themes/X --stages validate

# Skip regen if raw files exist, run the rest
python3 pipeline/run.py --config run_X.yaml --theme themes/X --no-regen

# Dataset + train only (after generation + validation done)
python3 pipeline/run.py --config run_X.yaml --theme themes/X --stages dataset train

# Preview without running
python3 pipeline/run.py --config run_X.yaml --theme themes/X --dry-run

# Skip HF mixing
python3 pipeline/run.py --config run_X.yaml --theme themes/X --no-hf

# Single generator manually
python3 pipeline/generate.py --config run_X.yaml --theme themes/X --model writer

# Single validation pass manually
python3 pipeline/validate.py --config run_X.yaml --theme themes/X \
    --pass fast-filter --source writer

python3 pipeline/validate.py --config run_X.yaml --theme themes/X \
    --pass cross-review --source writer --reviewer qwen
```
