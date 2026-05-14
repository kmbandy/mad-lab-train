# mad-lab-train v2 — API Contract & Design Spec

This document is the source of truth for the mad-lab-train v2 pipeline server and the Training Central Command dashboard tab. Both sides build against this spec. Do not begin implementation on either until this document is agreed upon.

**Server:** mad-lab-train FastAPI pipeline server, port 18840 (18820 is mcp-atlassian)  
**Client:** mad-lab-dash Training Central Command tab  
**Database:** Local PostgreSQL (same instance the dashboard already uses)  
**Migrations:** Alembic — all schema changes are versioned migration files  

---

## Table of Contents

1. [Database Schema](#1-database-schema)
2. [Alembic Migrations](#2-alembic-migrations)
3. [Log Retention Policy](#3-log-retention-policy)
4. [REST API](#4-rest-api)
5. [SSE Event Schema](#5-sse-event-schema)
6. [Job YAML Format](#6-job-yaml-format)
7. [Template Format](#7-template-format)
8. [Stage Config Schemas](#8-stage-config-schemas)

---

## 1. Database Schema

All pipeline state lives in PostgreSQL. Raw training output goes to files on disk (see §3). The `run_id` field is the primary trace key — it appears in every table so any piece of the system can be joined back to the originating run.

### 1.1 Enums

```sql
CREATE TYPE job_status AS ENUM (
    'pending', 'queued', 'running', 'paused',
    'completed', 'failed', 'cancelled'
);

CREATE TYPE stage_status AS ENUM (
    'pending', 'running', 'paused', 'completed', 'failed', 'skipped'
);

CREATE TYPE stage_type AS ENUM (
    'dataset_prep', 'data_gen', 'finetune', 'pretrain',
    'quant', 'merge', 'prune', 'eval', 'convert', 'upload'
);

CREATE TYPE execution_target AS ENUM ('local', 'ec2');
```

### 1.2 runs

The top-level job record. One row per job submission.

```sql
CREATE TABLE runs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name              TEXT NOT NULL,
    template_name     TEXT NOT NULL,               -- template used, or 'custom'
    status            job_status NOT NULL DEFAULT 'pending',
    execution_target  execution_target NOT NULL DEFAULT 'local',
    ec2_config        JSONB,                        -- null if local
    priority          INTEGER NOT NULL DEFAULT 100, -- lower = higher priority
    scheduled_for     TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    queued_at         TIMESTAMPTZ,
    started_at        TIMESTAMPTZ,
    ended_at          TIMESTAMPTZ,
    retain_logs_until TIMESTAMPTZ NOT NULL,         -- computed at creation, see §3
    error             TEXT
);

CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created_at ON runs(created_at DESC);
```

### 1.3 stages

One row per stage per run. Ordered by `sequence`.

```sql
CREATE TABLE stages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sequence    INTEGER NOT NULL,                  -- 0-indexed execution order
    stage_type  stage_type NOT NULL,
    status      stage_status NOT NULL DEFAULT 'pending',
    input_path  TEXT,                              -- auto-wired from previous stage
    output_path TEXT,                              -- set on completion
    started_at  TIMESTAMPTZ,
    ended_at    TIMESTAMPTZ,
    error       TEXT,
    UNIQUE (run_id, sequence)
);

CREATE INDEX idx_stages_run_id ON stages(run_id);
```

### 1.4 stage_configs

Stage configuration stored separately so large multi-stage jobs don't bloat the runs or stages rows. One row per stage.

```sql
CREATE TABLE stage_configs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_id    UUID NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
    stage_type  stage_type NOT NULL,
    config      JSONB NOT NULL,
    UNIQUE (stage_id)
);

CREATE INDEX idx_stage_configs_run_id ON stage_configs(run_id);
```

### 1.5 checkpoints

One row per checkpoint saved during execution.

```sql
CREATE TABLE checkpoints (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_id      UUID NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
    sequence      INTEGER NOT NULL,               -- monotonically increasing per stage
    is_clean      BOOLEAN NOT NULL DEFAULT TRUE,  -- false = force-paused
    artifact_path TEXT NOT NULL,                  -- directory of saved checkpoint files
    metadata      JSONB NOT NULL DEFAULT '{}',    -- stage-specific, see §1.7
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (stage_id, sequence)
);

CREATE INDEX idx_checkpoints_run_id ON checkpoints(run_id);
CREATE INDEX idx_checkpoints_stage_id ON checkpoints(stage_id);
```

### 1.6 events

Structured pipeline events. High-frequency events (e.g. training steps) are all stored here; raw unstructured output goes to log files. Rows older than `runs.retain_logs_until` are purged by the cleanup job.

```sql
CREATE TABLE events (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    stage_id    UUID REFERENCES stages(id) ON DELETE CASCADE,
    stage_type  stage_type,
    event_type  TEXT NOT NULL,
    data        JSONB NOT NULL DEFAULT '{}',
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_events_run_id ON events(run_id);
CREATE INDEX idx_events_run_id_ts ON events(run_id, ts DESC);
CREATE INDEX idx_events_stage_id ON events(stage_id);
```

### 1.7 templates

Pre-built and user-saved chain templates. Pre-built templates are seeded via Alembic migrations (see §2). User-saved templates are inserted at runtime.

```sql
CREATE TABLE templates (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT UNIQUE NOT NULL,              -- slug, no spaces
    label       TEXT NOT NULL,                     -- display name
    description TEXT NOT NULL DEFAULT '',
    chain       JSONB NOT NULL,                    -- ordered list of stage type + defaults
    is_builtin  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 1.8 Checkpoint Metadata by Stage Type

The `metadata` JSONB column in `checkpoints` contains stage-specific fields:

| stage_type | metadata fields |
|------------|----------------|
| `finetune` / `pretrain` | `step`, `epoch`, `train_loss`, `eval_loss` |
| `data_gen` | `samples_completed`, `total_samples` |
| `dataset_prep` | `sources_completed`, `total_sources`, `records_written` |
| `quant` | `quant_types_completed`, `current_quant_type` |
| `prune` | `layers_pruned`, `total_layers` |
| `eval` | `benchmarks_completed`, `total_benchmarks` |

---

## 2. Alembic Migrations

All schema changes are version-controlled as Alembic migration files in `mad-lab-train/alembic/versions/`. This is the source of truth for the database schema — no manual DDL, no undocumented changes.

### 2.1 File Structure

```
mad-lab-train/
  alembic/
    env.py
    script.py.mako
    versions/
      0001_initial_schema.py          -- enums + all tables
      0002_seed_builtin_templates.py  -- 6 pre-built templates
      0003_...                        -- future changes
  alembic.ini
```

### 2.2 Usage

```bash
# Apply all pending migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1

# Create a new migration
alembic revision -m "add_eval_results_table"

# Check current state
alembic current
```

### 2.3 Template Migrations

Pre-built templates are seeded in migration `0002_seed_builtin_templates.py` as `INSERT INTO templates` statements. This means:

- Template additions → new migration with `INSERT`
- Template edits → new migration with `UPDATE`
- Template removals → new migration with `DELETE WHERE is_builtin = TRUE AND name = '...'`

Every template change has a git commit, a migration number, and is fully reversible via `downgrade()`.

Example migration structure:

```python
# 0002_seed_builtin_templates.py

def upgrade():
    op.execute("""
        INSERT INTO templates (name, label, description, chain, is_builtin) VALUES
        ('gguf_quant', 'GGUF Quant',
         'Dataset prep and GGUF quantization with optional imatrix',
         '[{"stage_type": "dataset_prep", "defaults": {...}},
           {"stage_type": "quant", "defaults": {...}}]',
         TRUE),
        ('full_finetune', 'Full Finetune',
         'Dataset prep → data generation → QLoRA finetune → GGUF quant',
         '[{"stage_type": "dataset_prep", "defaults": {...}},
           {"stage_type": "data_gen", "defaults": {...}},
           {"stage_type": "finetune", "defaults": {...}},
           {"stage_type": "quant", "defaults": {...}}]',
         TRUE)
        -- ... remaining 4 templates
    """)

def downgrade():
    op.execute("DELETE FROM templates WHERE is_builtin = TRUE")
```

### 2.4 Pre-built Templates (seeded in 0002)

| name | label | chain |
|------|-------|-------|
| `gguf_quant` | GGUF Quant | dataset_prep → quant |
| `full_finetune` | Full Finetune | dataset_prep → data_gen → finetune → quant |
| `from_scratch` | From Scratch | dataset_prep → data_gen → pretrain → quant |
| `prune` | Prune | dataset_prep → data_gen → prune → finetune (healing) |
| `merge` | Merge | dataset_prep → data_gen → merge → eval → prune → finetune (healing) |
| `eval_standalone` | Eval Standalone | eval |

---

## 3. Log Retention Policy

### 3.1 Structured Events (Postgres)

All structured events are stored in the `events` table. `runs.retain_logs_until` is computed at job creation based on stage types in the run.

**Retention by stage type (after job ends):**

| Stage Type(s) in Run | Retention After Completion |
|----------------------|---------------------------|
| `pretrain` (From Scratch) | 14 days |
| `finetune`, `prune` | 14 days |
| `data_gen`, `dataset_prep` | 7 days |
| `quant`, `merge`, `convert` | 7 days |
| `eval`, `upload` | 3 days |

When a run contains multiple stage types, the longest applicable retention window wins.

**Hard rules (override all thresholds):**
- Never purge events for a run with status `running` or `paused`
- Never purge events for a run where `ended_at IS NULL`
- `retain_logs_until` is recomputed and extended on every checkpoint, so a long-running From Scratch job that takes 3 weeks never loses its logs mid-run

**Cleanup job:** runs daily at 3 AM via systemd timer.

```sql
-- Cleanup query (run daily)
DELETE FROM events
WHERE run_id IN (
    SELECT id FROM runs
    WHERE retain_logs_until < now()
    AND status NOT IN ('running', 'paused')
    AND ended_at IS NOT NULL
);
```

### 3.2 Raw Log Files (Disk)

Unstructured training output (full stdout/stderr from training processes) written to:

```
~/.mad-lab-train/logs/{run_id}/{stage_id}.log
```

These are tailed by the dashboard log panel. Cleaned up on the same schedule as events — the cleanup job deletes the log directory for a run when its `retain_logs_until` threshold passes.

---

## 4. REST API

Base URL: `http://localhost:18840`

All responses are JSON. Errors return `{"error": "message"}` with appropriate HTTP status.

### 4.1 Runs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/runs` | Create and queue a run |
| `GET` | `/runs` | List all runs |
| `GET` | `/runs/{id}` | Get full run detail (includes stages + configs) |
| `PATCH` | `/runs/{id}` | Update run config (queued status only) |
| `DELETE` | `/runs/{id}` | Delete run (queued or terminal state only) |

**POST /runs — Request**
```json
{
  "name":              "string",
  "template_name":     "string",
  "execution_target":  "local | ec2",
  "ec2_config":        "EC2Config | null",
  "stages":            "[{stage_type, config}]",
  "scheduled_for":     "ISO8601 | null",
  "start_immediately": "bool (default false)",
  "set_as_next":       "bool (default false)"
}
```

**POST /runs — Response: 201**
```json
{ "run": "Run" }
```

**GET /runs — Query params:** `status`, `limit` (default 50), `offset` (default 0)

**GET /runs/{id} — Response: 200**
```json
{
  "run": "Run",
  "stages": "[Stage]",
  "configs": "[StageConfig]"
}
```

### 4.2 Run Control

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/runs/{id}/start` | Start a queued run immediately |
| `POST` | `/runs/{id}/cancel` | Cancel a running or queued run |
| `POST` | `/runs/{id}/pause` | Clean pause (runs to next checkpoint then stops) |
| `POST` | `/runs/{id}/force-pause` | Immediate stop — resumes from previous clean checkpoint |
| `POST` | `/runs/{id}/resume` | Resume a paused run |
| `POST` | `/runs/{id}/priority` | Set queue priority |
| `POST` | `/runs/{id}/schedule` | Set or update scheduled start time |

**POST /runs/{id}/resume — Request**
```json
{ "checkpoint_id": "uuid | null (null = most recent clean checkpoint)" }
```

**POST /runs/{id}/priority — Request**
```json
{ "priority": "int" }
```

**POST /runs/{id}/schedule — Request**
```json
{ "scheduled_for": "ISO8601 | null (null clears schedule)" }
```

### 4.3 Streams, Logs & Checkpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/runs/{id}/stream` | SSE event stream (live) |
| `GET` | `/runs/{id}/logs` | Raw log tail |
| `GET` | `/runs/{id}/checkpoints` | List all checkpoints |

**GET /runs/{id}/logs — Query params:** `lines` (default 100), `stage_id` (optional filter)

**GET /runs/{id}/checkpoints — Response: 200**
```json
{ "checkpoints": "[Checkpoint]" }
```

### 4.4 Templates

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/templates` | List all templates (builtin + custom) |
| `GET` | `/templates/{name}` | Get template detail |
| `POST` | `/templates` | Save a custom template |
| `DELETE` | `/templates/{name}` | Delete custom template (builtin protected) |

**POST /templates — Request**
```json
{
  "name":        "string (slug)",
  "label":       "string",
  "description": "string",
  "chain":       "[{stage_type, defaults}]"
}
```

### 4.5 Hardware & Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/hardware` | Current hardware stats |
| `GET` | `/health` | Server health + DB connectivity check |

**GET /hardware — Response: 200**
```json
{
  "gpus": [
    {
      "index":           "int",
      "name":            "string",
      "vram_used_gb":    "float",
      "vram_total_gb":   "float",
      "utilization_pct": "int"
    }
  ],
  "cpu_pct":      "float",
  "ram_used_gb":  "float",
  "ram_total_gb": "float",
  "disk_free_gb": "float"
}
```

**GET /health — Response: 200**
```json
{
  "status":   "ok",
  "version":  "2.0.0",
  "db":       "connected | error"
}
```

### 4.6 EC2

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ec2/launch` | Launch spot instance for a run |
| `GET` | `/ec2/spot-price` | Current spot price for instance type |

**GET /ec2/spot-price — Query params:** `instance_type`, `region`

---

## 4.7 Queue & Scheduling Model

### States

| Status | Meaning |
|--------|---------|
| `pending` | Submitted, sitting in list, not queued for execution |
| `queued` | At the front of the queue (priority 0), waiting for manual start |
| `running` | Actively executing |
| `paused` | Paused (clean or force) — waiting for resume |
| `completed` | Finished successfully |
| `failed` | Finished with error |
| `cancelled` | Manually cancelled |

### Queue Rules

- **Manual start always required.** No run ever starts automatically, including scheduled runs and queue-front runs. The queue is a priority ordering only — not an auto-executor.
- **Priority:** lower integer = higher priority. Default is `100`. `set_as_next: true` sets `priority: 0`.
- **Tie-breaking:** priority value → `created_at` ascending (earlier submission wins). If two runs share identical priority and identical `created_at` to the millisecond → `409 Conflict`.
- **`scheduled_for`:** when the scheduled time arrives, the run's priority is set to `0` (moves to front of queue). Still requires manual start. If nothing is running at scheduled time, it remains `pending` at priority 0 — it does not auto-start.
- **`scheduled_for` and `set_as_next` are mutually exclusive** — both mean "move to front of queue," just different triggers. Supplying both → `400 Bad Request`.
- Only one run can have `priority: 0` at a time. Setting a new run as next does not error — it sets the existing priority-0 run back to `priority: 1` automatically.

### Dashboard Queue Actions

From the Status tab a queued (pending) run supports:
- **Start** — starts the run immediately (manual trigger)
- **Set as Next** — moves to priority 0
- **Schedule** — set a `scheduled_for` datetime (clears `set_as_next` if set)
- **Edit Config** — open config form (only available while status = `pending` or `queued`)
- **Cancel** — remove from queue

---

## 5. SSE Event Schema

All events are newline-delimited JSON on the `/runs/{id}/stream` endpoint. All events are also written to the `events` table.

### 5.1 Base Event Shape

```json
{
  "run_id":     "uuid",
  "stage_id":   "uuid | null",
  "stage_type": "stage_type | null",
  "event_type": "string",
  "ts":         "ISO8601",
  "data":       "object"
}
```

### 5.2 Run-Level Events

| event_type | data |
|------------|------|
| `run_started` | `{}` |
| `run_completed` | `{}` |
| `run_failed` | `{"error": "string"}` |
| `run_cancelled` | `{}` |
| `run_paused` | `{"checkpoint_id": "uuid", "is_clean": bool}` |
| `run_resumed` | `{"checkpoint_id": "uuid"}` |
| `pause_requested` | `{}` |
| `force_pause_requested` | `{}` |

### 5.3 Stage-Level Events (all stage types)

| event_type | data |
|------------|------|
| `stage_started` | `{"stage_type": "string", "sequence": int}` |
| `stage_completed` | `{"output_path": "string \| null"}` |
| `stage_failed` | `{"error": "string"}` |
| `stage_skipped` | `{"reason": "string"}` |
| `checkpoint` | `{"checkpoint_id": "uuid", "sequence": int, "metadata": object}` |

### 5.4 Finetune / Pretrain

| event_type | data |
|------------|------|
| `step` | `{"step": int, "total_steps": int, "epoch": int, "total_epochs": int, "loss": float, "lr": float, "grad_norm": float}` |
| `eval` | `{"step": int, "eval_loss": float, "perplexity": float}` |
| `epoch_end` | `{"epoch": int, "train_loss": float, "eval_loss": float}` |
| `vram` | `{"used_gb": float, "total_gb": float}` |
| `tokenizer_trained` | `{"vocab_size": int}` *(pretrain only)* |

### 5.5 Data Generation

| event_type | data |
|------------|------|
| `sample_generated` | `{"count": int, "total": int, "category": "string"}` |
| `sample_filtered` | `{"reason": "string", "score": float}` |
| `quality_score` | `{"score": float, "threshold": float, "passed": bool}` |

### 5.6 Dataset Prep

| event_type | data |
|------------|------|
| `source_started` | `{"source_name": "string", "source_type": "string"}` |
| `records_processed` | `{"count": int, "source_name": "string"}` |
| `source_complete` | `{"source_name": "string", "total_records": int}` |

### 5.7 Quantization

| event_type | data |
|------------|------|
| `quant_started` | `{"quant_type": "string"}` |
| `quant_progress` | `{"quant_type": "string", "percent": float}` |
| `quant_complete` | `{"quant_type": "string", "output_path": "string", "size_gb": float}` |

### 5.8 Prune

| event_type | data |
|------------|------|
| `importance_scored` | `{"method": "string"}` |
| `layer_pruned` | `{"layer_idx": int, "total_layers": int, "params_removed": int}` |

### 5.9 Evaluation

| event_type | data |
|------------|------|
| `benchmark_started` | `{"name": "string"}` |
| `sample_evaluated` | `{"count": int, "total": int}` |
| `benchmark_complete` | `{"name": "string", "score": float, "metric": "string"}` |
| `gate_result` | `{"passed": bool, "score": float, "threshold": float}` |

### 5.10 Merge / Convert / Upload

| event_type | data |
|------------|------|
| `merge_complete` | `{"output_path": "string"}` |
| `convert_complete` | `{"output_path": "string", "format": "string"}` |
| `upload_progress` | `{"bytes_sent": int, "total_bytes": int}` |
| `upload_complete` | `{"repo_url": "string"}` |

### 5.11 EC2

| event_type | data |
|------------|------|
| `ec2_instance_requested` | `{"instance_type": "string"}` |
| `ec2_instance_running` | `{"instance_id": "string", "ip": "string"}` |
| `ec2_instance_terminated` | `{"instance_id": "string", "reason": "string"}` |

---

## 6. Job YAML Format

A single YAML file per run. Used for job submission and config storage (the parsed result is stored in `stage_configs`). Omit any stage section to skip it.

```yaml
meta:
  name: "string"
  template: "string"

execution:
  target: local                    # local | ec2
  ec2:                             # required if target: ec2
    instance_type: g7e.2xlarge
    max_spot_price: "1.00"
    ami: ami-03bda78a7c7c13b45
    region: us-east-2
    key_name: mad-lab-key
    iam_profile: arn:aws:iam::080869524552:instance-profile/mad-lab-ec2-quant-role
    vpc_id: vpc-0a8e766998c7d5e23
    storage_gb: 300

stages:
  - type: dataset_prep
    config:
      sources:
        - type: huggingface
          repo: "dataset/name"
          split: train
          max_samples: 5000
        - type: zim
          path: /mnt/zim/wikipedia.zim
          query: "GPU architecture"
          max_records: 1000
        - type: qdrant
          url: "http://localhost:6333"
          collection: memory
          query: "training examples"
          top_k: 500
        - type: duckdb
          path: ~/.mneme/claude__main.db
          query: "SELECT content FROM facts WHERE type = 'fact' LIMIT 1000"
        - type: raw
          path: ~/datasets/custom.jsonl
      train_split: 0.9
      deduplicate: true

  - type: data_gen
    config:
      model: "unsloth/Qwen3-30B-A3B-GGUF"    # HF model ID — pulled by each worker
      system_prompt: "You are a helpful assistant."
      user_template: "Answer this question: {{ question }}"
      samples: 10000
      temperature: 0.85
      max_tokens: 512
      ctx_size: 2048
      quality_threshold: 0.7
      judge_model: ~                           # null = skip quality judging
      workers:
        - type: local
          host: mad-lab-main
          port: 8080
          parallel: 64
        - type: local
          host: mad-lab
          port: 8080
          parallel: 64
        - type: ec2                            # optional — omit to skip EC2 spoke
          instance_type: g6.4xlarge
          max_spot_price: "0.80"
          parallel: 300
          llama_cpp_s3: "s3://mad-lab/llama-cpp/llama-cpp-linux-cuda-b5000.tar.gz"

  - type: finetune
    config:
      mode: standard               # standard | healing
      base_model: /path/to/model
      gpu_target: auto             # auto | r9700 | 6900xt (local only; ignored on EC2)
      lora:
        r: 16
        alpha: 32
        dropout: 0.05
        target_modules: all-linear
      training:
        epochs: 3
        micro_batch_size: 1
        gradient_accumulation_steps: 16
        learning_rate: 2.0e-4
        lr_scheduler: cosine
        warmup_steps: 20
        weight_decay: 0.01
        max_grad_norm: 1.0
        gradient_checkpointing: false
        fp16: false
        bf16: true
        eval_steps: 100
        save_steps: 100
        logging_steps: 10

  - type: pretrain
    config:
      architecture: /path/to/arch.json
      vocab_size: 32000
      multi_gpu: false             # true = DeepSpeed ZeRO-2 across R9700+6900XT
      deepspeed_zero_stage: 2     # 1|2|3 — only used when multi_gpu: true
      gpu_target: auto             # auto | r9700 | 6900xt — only used when multi_gpu: false
      training:
        # same fields as finetune.training

  - type: quant
    config:
      model_path: /path/to/model   # omit to auto-wire from previous stage
      output_prefix: my-model
      quant_types: [Q4_K_M, Q5_K_M, TQ3_1S]
      imatrix: false
      imatrix_dataset: ~           # required if imatrix: true; auto-wired if dataset_prep present

  - type: merge
    config:
      base_model: /path/to/base
      adapter_path: ~              # auto-wired from finetune output
      eval_gate:
        enabled: true
        benchmark: perplexity
        threshold: 15.0
        on_fail: pause             # pause | abort

  - type: prune
    config:
      model_source: huggingface    # huggingface | local
      model_id: "org/model-name"   # required if model_source: huggingface
      model_path: ~                # required if model_source: local (or auto-wired from merge)
      method: wanda                # wanda (default) | llm_pruner
      pruning_ratio: 0.2
      calibration_dataset: ~       # auto-wired from dataset_prep
      llm_pruner:                  # only used when method: llm_pruner
        block_wise: true
        weight_pager: null         # future: NVMe→VRAM pager (WeightPager protocol)

  - type: eval
    config:
      model_path: ~                # auto-wired
      benchmarks:
        - type: perplexity
        - type: mmlu
          max_samples: 500
        - type: custom
          metric: exact_match

  - type: convert
    config:
      input_path: ~                # auto-wired
      format: gguf_f16             # gguf_f16 | gguf_bf16 | safetensors_fp32

  - type: upload
    config:
      source_path: ~               # auto-wired
      hf_repo: "org/model-name"
      visibility: public
      generate_model_card: true
```

---

## 7. Template Format

Templates are stored in the `templates` table. Pre-built templates are seeded via Alembic migration `0002_seed_builtin_templates.py`. User-saved templates are inserted at runtime via `POST /templates`.

The `chain` JSONB column holds an ordered array of stage definitions with their defaults:

```json
[
  {
    "stage_type": "dataset_prep",
    "defaults": {
      "train_split": 0.9,
      "deduplicate": true
    }
  },
  {
    "stage_type": "finetune",
    "defaults": {
      "mode": "standard",
      "lora": { "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear" },
      "training": { "epochs": 3, "learning_rate": 2e-4 }
    }
  }
]
```

The wizard reads the template's `chain` array to determine how many steps to show and pre-populate field defaults.

---

## 8. Stage Config Schemas

Full field reference for each stage type. All fields stored as JSONB in `stage_configs.config`.

### 8.1 dataset_prep

Output is a directory of tagged JSONL files consumed by downstream stages:
- `training.jsonl` — ChatML messages format, consumed by `finetune` and `data_gen` (as few-shot pool)
- `context.jsonl` — reference/seed docs, consumed by `data_gen` (randomly sampled per generation slot)
- `calibration.jsonl` — plain text records, consumed by `quant` imatrix and `prune` calibration only

All records are ChatML messages format. `calibration` records have no assistant turn. Consumers filter by file.

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `sources` | `list[SourceConfig]` | — | yes |
| `train_split` | `float` | `0.9` | no |
| `deduplicate` | `bool` | `true` | no |

**SourceConfig — shared fields:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `huggingface \| zim \| qdrant \| duckdb \| raw \| claude_jsonl` | — | yes |
| `purpose` | `training \| context \| calibration` | `training` | no |
| `max_records` | `int \| null` | `null` | no |

**SourceConfig — `huggingface`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `repo` | `string` | — | yes |
| `split` | `string` | `train` | no |
| `schema.format` | `instruction_response \| qa \| messages \| text` | — | yes |
| `schema.instruction_col` | `string` | `instruction` | if format=instruction_response |
| `schema.response_col` | `string` | `response` | if format=instruction_response |
| `schema.question_col` | `string` | `question` | if format=qa |
| `schema.answer_col` | `string` | `answer` | if format=qa |
| `schema.system_col` | `string \| null` | `null` | no |
| `schema.system_prompt` | `string \| null` | `null` | no |

**SourceConfig — `zim`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `path` | `string` | — | yes |
| `query` | `string \| null` | `null` (all articles) | no |

**SourceConfig — `qdrant`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `url` | `string` | — | yes |
| `collection` | `string` | — | yes |
| `query` | `string` | — | yes |
| `top_k` | `int` | `500` | no |

**SourceConfig — `duckdb`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `path` | `string` | — | yes |
| `query` | `string` | — | yes |
| `content_col` | `string` | `content` | no |

**SourceConfig — `raw`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `path` | `string` | — | yes |
| `schema.format` | `messages \| text \| instruction_response \| qa` | `messages` | no |

**SourceConfig — `claude_jsonl`:**

Mines Claude Code (or compatible agent) JSONL session transcripts into MAD-162-shaped trace records for memory-conditioned routing training (MAD-161). Each emitted record includes a top-level `messages` list (pipeline-compat) and a `trace` sidecar with `memory_calls`, session metadata, and provenance.

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `path` | `string` | — | yes (file or directory) |
| `recursive` | `bool` | `true` | no (directory mode only) |
| `unit` | `turn \| session` | `turn` | no |
| `min_turn_chars` | `int` | `0` | no |
| `agent` | `string` | `claude-code` | no (provenance tag) |
| `trace_source` | `organic_work \| organic_work_with_replay_injection \| programmatic_task \| benchmark \| seeded_synthetic \| captured_hook_injection` | `organic_work` | no |
| `reconstruct_injections` | `bool` | `false` | no — when `true`, each turn is enriched by querying the personal-KG `/context` endpoint with the user prompt (mirroring the runtime hook) and attaching the result as a synthetic `personal-kg-context-replay` memory_call. Trace_source auto-promotes to `organic_work_with_replay_injection`. |
| `kg_url` | `string` | `http://100.102.191.30:18830/context` | no (reconstruct_injections only) |
| `kg_timeout_s` | `float` | `3.0` | no (reconstruct_injections only) |
| `annotate_spans` | `bool` | `false` | no — when `true`, applies span-annotation tiers (verbatim → paraphrase → reasoning) to the assistant output. Span lists populated on the record's `trace`. |
| `span_min_match` | `int` | `30` | no (tier-1) — minimum char length for a verbatim retrieved span. |
| `span_max_blob_chars` | `int` | `200000` | no (tier-1) — cap on concatenated `memory_call.results` to keep difflib tractable. |
| `embedding_url` | `string \| null` | `null` | no (tier-2) — OpenAI-compatible /embeddings endpoint. When set with `embedding_model`, paraphrase detection runs after the verbatim pass. |
| `embedding_model` | `string \| null` | `null` | no (tier-2) — model name sent in the embeddings request payload. No default; must be provided. |
| `paraphrase_threshold` | `float 0–1` | `0.75` | no (tier-2) — min cosine similarity to mark an assistant sentence as a paraphrased retrieved span. |
| `paraphrase_min_chars` | `int` | `40` | no (tier-2) — sentence-fragment length floor for the paraphrase pass. |
| `embedding_timeout_s` | `float` | `10.0` | no (tier-2) |
| `labeler` | `LabelerConfig \| null` | `null` | no (tier-3) — when present, an LLM worker pool labels remaining generation spans as REASONING vs GENERATION. See LabelerConfig below. |

**LabelerConfig (tier-3 span annotation, reuses data_gen's WorkerPool):**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `workers` | `list[WorkerConfig]` | — | yes — local llama.cpp endpoints; see §8.2 WorkerConfig — local. No default model. |
| `model` | `string` | `""` | no — passed for endpoint validation; the actual model is whatever the server hosts. |
| `sample_rate` | `float 0–1` | `0.1` | no — fraction of records to label (deterministic per session_id+timestamp). |
| `quality_audit` | `int` | `100` | no — reserved; spot-check N labeled records for downstream review. |
| `temperature` | `float` | `0.0` | no |
| `max_tokens` | `int` | `64` | no |

### 8.2 data_gen

Data gen always runs as hub-and-spoke from mad-lab-main regardless of run `execution_target`. See §9.3.

**Context sampling:** when `context.jsonl` exists from a preceding `dataset_prep` stage, the coordinator randomly samples one context doc per generation slot and injects it into `user_template` as `{{ context }}`. Random sampling (not sequential) ensures variety across the full run and avoids topic drift. Each worker receives an independent random draw.

`user_template` is a Jinja2 template. Available variables: `{{ context }}` (random context doc if available), `{{ topic }}` (optional explicit topic list, round-robined).

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model` | `string` | — | yes |
| `system_prompt` | `string` | — | yes |
| `user_template` | `string` | — | yes |
| `samples` | `int` | `1000` | no |
| `temperature` | `float` | `0.85` | no |
| `max_tokens` | `int` | `512` | no |
| `ctx_size` | `int` | `2048` | no |
| `quality_threshold` | `float 0–1` | `0.7` | no |
| `judge_model` | `string \| null` | `null` | no |
| `topics` | `list[string] \| null` | `null` | no |
| `workers` | `list[WorkerConfig]` | — | yes |

**WorkerConfig — local:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `local` | — | yes |
| `host` | `string` | — | yes |
| `port` | `int` | `8080` | no |
| `parallel` | `int` | `50` | no |

**WorkerConfig — ec2:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `ec2` | — | yes |
| `instance_type` | `string` | — | yes |
| `max_spot_price` | `string` | — | yes |
| `parallel` | `int` | `300` | no |
| `llama_cpp_s3` | `string` | — | yes |

### 8.3 finetune

QLoRA via TRL SFTTrainer + PEFT. Always single-GPU. See §9.2.

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `mode` | `standard \| healing` | `standard` | no |
| `base_model` | `string` | — | yes |
| `gpu_target` | `auto \| r9700 \| 6900xt` | `auto` | no |
| `lora.r` | `int` | `16` | no |
| `lora.alpha` | `int` | `32` | no |
| `lora.dropout` | `float` | `0.05` | no |
| `lora.target_modules` | `string` | `all-linear` | no |
| `training.epochs` | `int` | `3` (healing: `1`) | no |
| `training.micro_batch_size` | `int` | `1` | no |
| `training.gradient_accumulation_steps` | `int` | `16` | no |
| `training.learning_rate` | `float` | `2e-4` (healing: `5e-4`) | no |
| `training.lr_scheduler` | `string` | `cosine` | no |
| `training.warmup_steps` | `int` | `20` | no |
| `training.weight_decay` | `float` | `0.01` | no |
| `training.max_grad_norm` | `float` | `1.0` | no |
| `training.gradient_checkpointing` | `bool` | `false` | no |
| `training.fp16` | `bool` | auto-detected | no |
| `training.bf16` | `bool` | auto-detected | no |
| `training.eval_steps` | `int` | `100` | no |
| `training.save_steps` | `int` | `100` | no |
| `training.logging_steps` | `int` | `10` | no |

### 8.4 pretrain

Full weight training. Supports DeepSpeed ZeRO-2 multi-GPU or single-GPU. See §9.2.

Same as `finetune` training fields plus:

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `architecture` | `string (path to arch JSON)` | — | yes |
| `vocab_size` | `int` | `32000` | no |
| `multi_gpu` | `bool` | `false` | no |
| `deepspeed_zero_stage` | `int 1\|2\|3` | `2` | no (multi_gpu only) |
| `gpu_target` | `auto \| r9700 \| 6900xt` | `auto` | no (single GPU only) |

### 8.5 quant

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model_path` | `string` | auto-wired | no |
| `output_prefix` | `string` | run name | no |
| `quant_types` | `list[string]` | `[Q4_K_M]` | no |
| `imatrix` | `bool` | `false` | no |
| `imatrix_dataset` | `string` | auto-wired if imatrix true | no |

### 8.6 merge

Two modes. `adapter` is auto-inserted by the executor when finetune → quant is detected and no explicit merge stage exists. Model-to-model modes use `mergekit` on CPU (no GPU required — SN850 NVMe handles I/O).

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `mode` | `adapter \| slerp \| ties \| dare_ties` | `adapter` | no |
| `base_model` | `string` | — | yes |
| `adapter_path` | `string` | auto-wired | adapter mode only |
| `model_b` | `string` | — | slerp/ties/dare_ties |
| `model_c` | `string \| null` | `null` | ties/dare_ties (optional 3rd) |
| `merge_ratio` | `float 0–1` | `0.5` | slerp only |
| `density` | `float 0–1` | `0.5` | dare_ties (drop fraction) |
| `eval_gate.enabled` | `bool` | `false` | no |
| `eval_gate.benchmark` | `string` | `perplexity` | no |
| `eval_gate.threshold` | `float` | — | if gate enabled |
| `eval_gate.on_fail` | `pause \| abort` | `pause` | no |

### 8.7 prune

Always operates on full-precision safetensors. Downloaded from HuggingFace at job time if `model_source: huggingface`. See §9.4.

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model_source` | `huggingface \| local` | `huggingface` | no |
| `model_id` | `string` | — | if model_source=huggingface |
| `model_path` | `string` | auto-wired | if model_source=local |
| `method` | `wanda \| llm_pruner` | `wanda` | no |
| `pruning_ratio` | `float 0–1` | `0.2` | no |
| `calibration_dataset` | `string` | auto-wired | no |
| `llm_pruner.block_wise` | `bool` | `true` | no (llm_pruner only) |
| `llm_pruner.weight_pager` | `string \| null` | `null` | no (llm_pruner only) |

### 8.8 eval

All benchmark types produce a normalized score (0–1, higher is better) so the eval gate thresholds uniformly regardless of benchmark type.

**Gate behaviors on failure:**
- `pause` — run enters `paused` state; user can adjust merge config and re-run the merge stage, override the gate and continue, or abort
- `abort` — run cancelled immediately

*Future enhancement (not v1):* automatic rollback to previous clean checkpoint with adjusted hyperparameters.

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model_path` | `string` | auto-wired | no |
| `benchmarks` | `list[BenchmarkConfig]` | — | yes |

**BenchmarkConfig — `perplexity`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `perplexity` | — | yes |
| `dataset` | `string` | auto-wired from calibration.jsonl | no |
| `max_samples` | `int` | `200` | no |

Score: `1 / (1 + perplexity)` — normalized so lower perplexity = higher score.

**BenchmarkConfig — `mmlu`:**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `mmlu` | — | yes |
| `max_samples` | `int \| null` | `null` (all) | no |
| `subjects` | `list[string] \| null` | `null` (all) | no |

Score: accuracy (0–1).

**BenchmarkConfig — `tool_use` (stub):**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `tool_use` | — | yes |
| `dataset` | `string` | — | yes |
| `judge_model` | `string \| null` | `null` (exact match only) | no |

Dataset JSONL schema: `{"prompt": "...", "tools": [...], "expected_tool": "...", "expected_params": {...}}`

Score: tool selection + parameter accuracy (0–1). *Execution stubbed in v1.*

**BenchmarkConfig — `conversation` (stub):**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `conversation` | — | yes |
| `dataset` | `string` | — | yes |
| `judge_model` | `string` | — | yes |

Dataset JSONL schema: `{"turns": [{"role": "...", "content": "..."}], "reference_response": "..."}`

Score: LLM-as-judge coherence + helpfulness (0–1). *Execution stubbed in v1.*

**BenchmarkConfig — `coding` (stub):**

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `type` | `coding` | — | yes |
| `dataset` | `string` | — | yes |
| `language` | `string` | `python` | no |
| `timeout_seconds` | `int` | `10` | no |

Dataset JSONL schema: `{"prompt": "...", "language": "...", "test_code": "..."}`

Score: pass@1 rate (0–1). *Execution stubbed in v1.*

### 8.9 convert

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `input_path` | `string` | auto-wired | no |
| `format` | `gguf_f16 \| gguf_bf16 \| safetensors_fp32` | `gguf_f16` | no |

### 8.10 upload

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `source_path` | `string` | auto-wired | no |
| `hf_repo` | `string` | — | yes |
| `visibility` | `public \| private` | `public` | no |
| `generate_model_card` | `bool` | `true` | no |

---

## 9. Compute Architecture

### 9.1 Machine Topology

| Host | Role | GPUs | GPU Framework | Notes |
|------|------|------|---------------|-------|
| `mad-lab-main` | coordinator + trainer | 2 (PCIe) | ROCm | 3900X 12c/24t, SN850 NVMe, 16GB RAM |
| `mad-lab` | inference spoke | 2 (PCIe) | ROCm | training GPUs not powerful enough for multi-GPU overhead |
| EC2 spot | training or inference spoke | varies | CUDA | launched on-demand via boto3 |

The R9700 is connected via PCIe on the motherboard. The RX 6900 XT is in an eGPU enclosure connected via Thunderbolt 3 (~3-4 GB/s effective GPU-to-GPU bandwidth).

Total local VRAM: 48 GB across both GPUs on mad-lab-main.
System RAM on mad-lab-main: 16 GB (limits CPU offload ceiling for large model pruning).

### 9.2 Training Framework by Stage Type

#### Pretrain (From Scratch)

- **Multi-GPU:** DeepSpeed ZeRO-2 with `overlap_comm: true` — overlaps gradient sync with backward pass to hide TB3 latency
- **Single-GPU:** target either GPU explicitly (`gpu_target: r9700 | 6900xt | auto`)
- **Single-GPU default:** recommended when model fits — avoids TB3 overhead entirely
- **EC2:** CUDA, DeepSpeed or single-GPU depending on instance type

#### Finetune (QLoRA)

- **Always single-GPU** — QLoRA's 4-bit base + LoRA adapters fit on one card; DeepSpeed + bitsandbytes has known compatibility friction
- `gpu_target: r9700 | 6900xt | auto` — enables parallel runs (e.g. healing finetune on 6900XT while something else runs on R9700)
- `mode: healing` reuses the same executor with higher LR and fewer epochs — no separate executor needed
- ROCm locally, CUDA on EC2

### 9.3 Data Generation — Hub-and-Spoke

Data generation always runs as a distributed hub-and-spoke job regardless of the run's `execution_target`.

**Coordinator (always mad-lab-main):**
- Python orchestration process runs on the 3900X — handles batching, fan-out, result collection, deduplication, quality scoring, and file writes
- All output written to mad-lab-main SN850 NVMe (`~/.mad-lab-train/datagen/{run_id}/`)

**Spokes (inference endpoints):**
- Each spoke is a llama.cpp server with `--parallel N`
- Coordinator distributes requests proportionally to each spoke's `parallel` capacity
- Active spokes can be any combination of: mad-lab-main local GPUs, mad-lab GPUs, EC2 instance

**Local spoke lifecycle (check-before-kill):**
1. Check if llama.cpp is already running on the target host+port
2. If running with matching model and args → leave it, register as spoke
3. If running with different model/args → kill it, start fresh
4. If not running → start it

**EC2 spoke lifecycle:**
1. Launch spot instance via boto3 (shared EC2 bootstrap module — see §9.5)
2. Pull llama.cpp binary from S3 cache (avoid recompile on every launch)
3. Pull model from HuggingFace using token from AWS SSM `/mad-lab/hf-token`
4. Start llama.cpp server with configured `--parallel` and `--ctx-size`
5. Wait for health check → register as spoke
6. Terminate instance when data_gen stage completes

**Parallelism targets:**
- Local spoke (per GPU): 50–100 parallel slots depending on model size and VRAM
- EC2 spoke: 250–500 parallel slots depending on instance type
- Total across all spokes can exceed 500+ simultaneous requests — coordinator must respect per-spoke capacity

### 9.4 Pruning Compute

Pruning always operates on full-precision safetensors (FP16/BF16) downloaded from HuggingFace at job time. GGUFs cannot be pruned. The correct chain is always:

```
HF safetensors → prune → healing finetune → quant to GGUF
```

**Method selection:**
- **Wanda (default):** processes one layer at a time, forward-pass only, minimal memory overhead. Uses `device_map="auto"` + explicit `max_memory` caps to spread across both local GPUs. Works well with models up to ~30B locally; beyond that, CPU offload to system RAM (16GB headroom is limited).
- **LLM-Pruner (optional):** structural pruning — removes entire attention heads and MLP channels based on dependency graphs. Requires forward+backward pass per group (2-3x memory vs Wanda). Uses `--block_wise` mode in v1. A `WeightPager` abstraction slot is reserved for future integration with the NVMe→VRAM weight pager being built for MoE routing (architecturally identical primitive).

**WeightPager interface (v1 stub, v2 implementation):**
```python
class WeightPager(Protocol):
    def page_in(self, layer_ids: list[str]) -> None: ...
    def page_out(self, layer_ids: list[str]) -> None: ...
```

LLM-Pruner executor accepts an optional `weight_pager: WeightPager | None`. When `None` (v1), falls back to `device_map` + CPU offload. When the MoE NVMe pager is built, it implements this protocol and can be injected with no changes to the pruning executor.

### 9.5 Shared EC2 Bootstrap Module

All stages that use EC2 (training, data_gen spoke, quant) share a single bootstrap module at `pipeline/ec2/bootstrap.py`. No stage duplicates EC2 launch/teardown logic.

Responsibilities:
- Launch spot instance via boto3 with run-specific config
- Fetch HF token from AWS SSM Parameter Store (`/mad-lab/hf-token`, `--with-decryption`)
- Pull stage-specific artifacts from S3 (llama.cpp binary, training scripts, etc.)
- Execute remote setup script via SSH
- Health-check the remote service
- Register instance ID on the run record (`ec2_config.instance_id`)
- Terminate on stage completion or failure

**EC2 config fields (shared across all stages):**

| Field | Description |
|-------|-------------|
| `instance_type` | e.g. `g6.4xlarge`, `g7e.2xlarge` |
| `max_spot_price` | USD/hr ceiling |
| `ami` | Base AMI ID |
| `region` | AWS region |
| `key_name` | EC2 key pair name |
| `iam_profile` | Instance profile ARN |
| `vpc_id` | VPC ID |
| `storage_gb` | Root volume size |
| `s3_artifacts` | List of S3 paths to pull on boot (stage-specific) |

---

## Auto-Wiring Rules

When stages are chained, outputs are automatically wired to inputs at job creation time. Explicit values in the config override auto-wiring.

| Producing Stage | Output | Consuming Stage | Auto-wired Field |
|----------------|--------|-----------------|-----------------|
| `dataset_prep` | `context.jsonl` | `data_gen` | context pool (random sampled) |
| `dataset_prep` | `training.jsonl` | `data_gen` | few-shot pool |
| `dataset_prep` | `training.jsonl` | `finetune` | `dataset` |
| `dataset_prep` | `training.jsonl` (eval split) | `finetune` | `eval_dataset` |
| `dataset_prep` | `calibration.jsonl` | `quant` | `imatrix_dataset` |
| `dataset_prep` | `calibration.jsonl` | `prune` | `calibration_dataset` |
| `dataset_prep` | `calibration.jsonl` | `eval` | benchmark dataset |
| `data_gen` | `generated.jsonl` | `finetune` | `dataset` *(merged with training.jsonl if both exist)* |
| `finetune` | adapter dir | `merge (adapter mode)` | `adapter_path` |
| `finetune` | adapter dir | `quant` | triggers auto-insert of `merge (adapter)` stage |
| `pretrain` | model dir | `quant` | `model_path` |
| `merge` | merged model dir | `prune` | `model_path` |
| `merge` | merged model dir | `eval` | `model_path` |
| `prune` | pruned model dir | `finetune` (healing) | `base_model` |
| `finetune` (after adapter merge) | merged weights dir | `quant` | `model_path` |

**Validation at job creation:**
- If `quant.imatrix: true` and no `dataset_prep` stage exists in the chain → 400 error
- If `merge.eval_gate.enabled: true` and `eval_gate.threshold` is not set → 400 error
- Stages must have a valid `sequence` order (no circular dependencies)

---

---

## 10. Dashboard UI — Training Central Command

### 10.1 Tab Structure

The Training Central Command occupies a single top-level tab in mad-dashboard. Inside it are two sub-tabs: **Status** and **New Job**.

### 10.2 Status Tab

**Run list (top):** table of all runs, newest first.

| Column | Notes |
|--------|-------|
| Name | run name |
| Template | template label |
| Model | primary model (from first stage config) |
| Status | color-coded pill |
| Target | `local` or `ec2` |
| Duration | elapsed or total time |
| Created | relative timestamp |

Click a row → expands detail panel below (or slide-in panel).

**Detail panel layout — split:**
- **Left:** Cytoscape chain graph — all stages as nodes, edges showing flow
- **Right:** tabbed panel with tabs: Overview, Config, Logs, Events

Default tab by status: running → Logs, completed/failed → Overview, paused → Overview.

**Overview tab:** run metadata (id, name, template, target, timing), stage timing table, error message if failed, gate failure details if paused.

**Config tab:** full stage configs rendered as read-only YAML-style view. If status is `pending` or `queued`, an Edit button opens the inline edit form.

**Logs tab:** live-tailing log output from `~/.mad-lab-train/logs/{run_id}/`. Stage selector dropdown when multiple stages have logs. Auto-scrolls to bottom; scroll up to pause auto-scroll.

**Events tab:** structured event stream from the `events` table, newest-first. Filterable by stage and event type.

**Queue action buttons** (visible based on status):

| Status | Available Actions |
|--------|------------------|
| `pending` | Start, Set as Next, Schedule, Edit Config, Cancel |
| `queued` | Start, Edit Config, Cancel |
| `running` | Pause, Force Pause, Cancel |
| `paused` (clean) | Resume, Cancel |
| `paused` (gate failure) | Adjust Config + Retry Merge, Override Gate, Abort |

**Notifications:** visual only. Tab title shows badge count of active runs. Status pills update live via SSE. No browser push notifications (notification framework deferred — larger play across agents + dashboard at scale).

### 10.3 Cytoscape Chain Visualization

Each stage is a node. Directed edges show execution order. Node appearance by status:

| Status | Color | Label |
|--------|-------|-------|
| `pending` | gray | stage type |
| `running` | yellow, pulsing | stage type + progress % |
| `completed` | green | stage type + ✓ |
| `failed` | red | stage type + ✗ |
| `skipped` | light gray | stage type + skipped |
| `paused` | amber | stage type + paused |

**Progress % per stage type** (fed from SSE events in real time):

| Stage | Event | Calculation |
|-------|-------|-------------|
| `finetune` / `pretrain` | `step` | `step / total_steps × 100` |
| `data_gen` | `sample_generated` | `count / total × 100` |
| `dataset_prep` | `source_complete` | `sources_done / total_sources × 100` |
| `quant` | `quant_complete` | `types_done / total_types × 100` |
| `prune` | `layer_pruned` | `layers_done / total_layers × 100` |
| `eval` | `benchmark_complete` | `benchmarks_done / total × 100` |
| `upload` | `upload_progress` | `bytes_sent / total_bytes × 100` |
| `merge` / `convert` | — | spinner only (no granular progress) |

The Cytoscape graph in the Status detail panel updates live. The same graph renders (static) in the New Job wizard sidebar during configuration.

### 10.4 New Job Wizard (Stepper)

**Step 0 — Template picker:**
Grid of template cards. Each card shows: icon, label, chain summary (e.g. `dataset_prep → data_gen → finetune → quant`). Includes a "Custom" card for manual chain building. Selecting a template renders the chain as a Cytoscape preview immediately.

**Step 1 — Run setup:**
- Run name
- Execution target: `local` | `ec2` (EC2 fields appear conditionally)
- Scheduling: none | set as next | schedule for datetime (mutually exclusive)

**Steps 2–N — One step per stage in the chain:**
- Step header = stage type label
- Form fields pre-populated from template defaults
- Required fields marked, optional fields collapsible
- Cytoscape chain graph in sidebar with current stage node pulsing

**Final step — Review:**
Full config summary (read-only). "Submit to Queue" button.

Inline edit (for pending/queued runs from Status tab): same form as the wizard steps, but accessed directly at the stage level — no full re-wizard flow needed.

---

*Last updated: 2026-05-02 — MAD-69, compute architecture + full design session*
