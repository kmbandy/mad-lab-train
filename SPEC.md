# mad-lab-train v2 — API Contract & Design Spec

This document is the source of truth for the mad-lab-train v2 pipeline server and the Training Central Command dashboard tab. Both sides build against this spec. Do not begin implementation on either until this document is agreed upon.

**Server:** mad-lab-train FastAPI pipeline server, port 18820  
**Client:** mad-lab-dash Training Central Command tab  

---

## Table of Contents

1. [Data Models](#1-data-models)
2. [REST API](#2-rest-api)
3. [SSE Event Schema](#3-sse-event-schema)
4. [Job YAML Format](#4-job-yaml-format)
5. [Template Format](#5-template-format)
6. [Checkpoint Model](#6-checkpoint-model)
7. [Stage Config Schemas](#7-stage-config-schemas)

---

## 1. Data Models

### 1.1 JobStatus

```
PENDING     — created, not yet queued
QUEUED      — in queue, waiting for execution (config still editable)
RUNNING     — actively executing
PAUSED      — stopped at a checkpoint, ready to resume
COMPLETED   — all stages finished successfully
FAILED      — a stage failed; job stopped
CANCELLED   — user-cancelled before completion
```

### 1.2 StageStatus

```
PENDING     — waiting for previous stage to complete
RUNNING     — actively executing
PAUSED      — stopped at checkpoint
COMPLETED   — finished successfully
FAILED      — errored out
SKIPPED     — conditionally skipped (e.g. imatrix disabled)
```

### 1.3 StageType

```
dataset_prep | data_gen | finetune | pretrain | quant | merge | prune | eval | convert | upload
```

Healing Finetune is not a separate stage type — it is `finetune` with `mode: healing` in its config.

### 1.4 Job

```json
{
  "id":               "uuid",
  "name":             "string",
  "template":         "string (template name used, or 'custom')",
  "status":           "JobStatus",
  "execution_target": "local | ec2",
  "ec2_config":       "EC2Config | null",
  "stages":           "[Stage]",
  "created_at":       "ISO8601",
  "queued_at":        "ISO8601 | null",
  "scheduled_for":    "ISO8601 | null",
  "started_at":       "ISO8601 | null",
  "ended_at":         "ISO8601 | null",
  "priority":         "int (lower = higher priority, default 100)",
  "error":            "string | null"
}
```

### 1.5 Stage

```json
{
  "id":           "uuid",
  "job_id":       "uuid",
  "sequence":     "int (0-indexed, defines execution order)",
  "stage_type":   "StageType",
  "status":       "StageStatus",
  "config":       "object (stage-specific, see §7)",
  "input_path":   "string | null (auto-wired from previous stage output_path)",
  "output_path":  "string | null (set on completion)",
  "started_at":   "ISO8601 | null",
  "ended_at":     "ISO8601 | null",
  "error":        "string | null"
}
```

### 1.6 EC2Config

```json
{
  "instance_type":  "string (e.g. g7e.2xlarge)",
  "max_spot_price": "string (e.g. '1.00')",
  "ami":            "string",
  "region":         "string (e.g. us-east-2)",
  "key_name":       "string",
  "iam_profile":    "string (ARN)",
  "vpc_id":         "string",
  "storage_gb":     "int"
}
```

---

## 2. REST API

Base URL: `http://localhost:18820`

All responses are JSON. Errors return `{"error": "message"}` with appropriate HTTP status.

### 2.1 Jobs

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Create and queue a job |
| `GET` | `/jobs` | List all jobs |
| `GET` | `/jobs/{id}` | Get full job detail |
| `PATCH` | `/jobs/{id}` | Update job config (QUEUED only) |
| `DELETE` | `/jobs/{id}` | Delete job (QUEUED or terminal state only) |

**POST /jobs — Request**
```json
{
  "name":             "string",
  "template":         "string",
  "execution_target": "local | ec2",
  "ec2_config":       "EC2Config | null",
  "stages":           "[StageConfig]",
  "scheduled_for":    "ISO8601 | null",
  "start_immediately": "bool (default false)",
  "set_as_next":      "bool (default false)"
}
```

**POST /jobs — Response: 201**
```json
{ "job": "Job" }
```

**GET /jobs — Query params:** `status`, `limit` (default 50), `offset` (default 0)  
**GET /jobs — Response: 200**
```json
{ "jobs": "[Job]", "total": "int" }
```

**GET /jobs/{id} — Response: 200**
```json
{ "job": "Job" }
```

**PATCH /jobs/{id} — Request** (QUEUED state only)
```json
{
  "name":             "string | omit",
  "execution_target": "local | ec2 | omit",
  "ec2_config":       "EC2Config | null | omit",
  "stages":           "[StageConfig] | omit"
}
```

### 2.2 Job Control

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs/{id}/start` | Start a QUEUED job immediately |
| `POST` | `/jobs/{id}/cancel` | Cancel a RUNNING or QUEUED job |
| `POST` | `/jobs/{id}/pause` | Request clean pause (runs to next checkpoint) |
| `POST` | `/jobs/{id}/force-pause` | Immediate stop (resumes from previous checkpoint) |
| `POST` | `/jobs/{id}/resume` | Resume a PAUSED job |
| `POST` | `/jobs/{id}/priority` | Set queue priority |
| `POST` | `/jobs/{id}/schedule` | Set or update scheduled start time |

**POST /jobs/{id}/resume — Request**
```json
{ "checkpoint_id": "uuid | null (null = most recent clean checkpoint)" }
```

**POST /jobs/{id}/priority — Request**
```json
{ "priority": "int" }
```

**POST /jobs/{id}/schedule — Request**
```json
{ "scheduled_for": "ISO8601 | null (null clears schedule)" }
```

### 2.3 Streams & Logs

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/jobs/{id}/stream` | SSE event stream (live) |
| `GET` | `/jobs/{id}/logs` | Raw log tail |
| `GET` | `/jobs/{id}/checkpoints` | List all checkpoints for a job |

**GET /jobs/{id}/logs — Query params:** `lines` (default 100), `stage_id` (optional filter)

**GET /jobs/{id}/checkpoints — Response: 200**
```json
{ "checkpoints": "[Checkpoint]" }
```

### 2.4 Templates

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/templates` | List all templates (pre-built + custom) |
| `GET` | `/templates/{name}` | Get template detail |
| `POST` | `/templates` | Save a custom template |
| `DELETE` | `/templates/{name}` | Delete a custom template (pre-built protected) |

**GET /templates — Response: 200**
```json
{
  "templates": [
    {
      "name":        "string",
      "label":       "string (display name)",
      "description": "string",
      "chain":       "[StageType]",
      "is_builtin":  "bool",
      "created_at":  "ISO8601 | null"
    }
  ]
}
```

**POST /templates — Request**
```json
{
  "name":        "string (slug, no spaces)",
  "label":       "string",
  "description": "string",
  "stages":      "[StageConfig]"
}
```

### 2.5 Hardware & Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/hardware` | Current hardware stats |
| `GET` | `/health` | Server health check |

**GET /hardware — Response: 200**
```json
{
  "gpu": [
    {
      "index":     "int",
      "name":      "string",
      "vram_used_gb": "float",
      "vram_total_gb": "float",
      "utilization_pct": "int"
    }
  ],
  "cpu_pct":    "float",
  "ram_used_gb": "float",
  "ram_total_gb": "float",
  "disk_free_gb": "float"
}
```

**GET /health — Response: 200**
```json
{ "status": "ok", "version": "2.0.0" }
```

### 2.6 EC2

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ec2/launch` | Launch spot instance for a job |
| `GET` | `/ec2/spot-price` | Current spot price for instance type |

**GET /ec2/spot-price — Query params:** `instance_type`, `region`  
**GET /ec2/spot-price — Response: 200**
```json
{ "instance_type": "string", "region": "string", "price_usd": "float" }
```

---

## 3. SSE Event Schema

All events are newline-delimited JSON on the `/jobs/{id}/stream` endpoint.

### 3.1 Base Event Shape

```json
{
  "job_id":     "uuid",
  "stage_id":   "uuid | null",
  "stage_type": "StageType | null",
  "event_type": "string",
  "timestamp":  "ISO8601",
  "data":       "object"
}
```

### 3.2 Job-Level Events

| event_type | data |
|------------|------|
| `job_started` | `{}` |
| `job_completed` | `{}` |
| `job_failed` | `{"error": "string"}` |
| `job_cancelled` | `{}` |
| `job_paused` | `{"checkpoint_id": "uuid", "is_clean": true}` |
| `job_resumed` | `{"checkpoint_id": "uuid"}` |
| `pause_requested` | `{}` |
| `force_pause_requested` | `{}` |

### 3.3 Stage-Level Events (all stage types)

| event_type | data |
|------------|------|
| `stage_started` | `{"stage_type": "string", "sequence": int}` |
| `stage_completed` | `{"output_path": "string \| null"}` |
| `stage_failed` | `{"error": "string"}` |
| `stage_skipped` | `{"reason": "string"}` |
| `checkpoint` | `{"checkpoint_id": "uuid", "sequence": int, "metadata": object}` |

### 3.4 Finetune / Pretrain Events

| event_type | data |
|------------|------|
| `step` | `{"step": int, "total_steps": int, "epoch": int, "total_epochs": int, "loss": float, "lr": float, "grad_norm": float}` |
| `eval` | `{"step": int, "eval_loss": float, "perplexity": float}` |
| `epoch_end` | `{"epoch": int, "train_loss": float, "eval_loss": float}` |
| `vram` | `{"used_gb": float, "total_gb": float}` |
| `tokenizer_trained` | `{"vocab_size": int}` (pretrain only) |

### 3.5 Data Generation Events

| event_type | data |
|------------|------|
| `sample_generated` | `{"count": int, "total": int, "category": "string"}` |
| `sample_filtered` | `{"reason": "string", "score": float}` |
| `quality_score` | `{"score": float, "threshold": float, "passed": bool}` |

### 3.6 Dataset Prep Events

| event_type | data |
|------------|------|
| `source_started` | `{"source_name": "string", "source_type": "string"}` |
| `records_processed` | `{"count": int, "source_name": "string"}` |
| `source_complete` | `{"source_name": "string", "total_records": int}` |

### 3.7 Quantization Events

| event_type | data |
|------------|------|
| `quant_started` | `{"quant_type": "string"}` |
| `quant_progress` | `{"quant_type": "string", "percent": float}` |
| `quant_complete` | `{"quant_type": "string", "output_path": "string", "size_gb": float}` |

### 3.8 Prune Events

| event_type | data |
|------------|------|
| `importance_scored` | `{"method": "string"}` |
| `layer_pruned` | `{"layer_idx": int, "total_layers": int, "params_removed": int}` |

### 3.9 Evaluation Events

| event_type | data |
|------------|------|
| `benchmark_started` | `{"name": "string"}` |
| `sample_evaluated` | `{"count": int, "total": int}` |
| `benchmark_complete` | `{"name": "string", "score": float, "metric": "string"}` |
| `gate_result` | `{"passed": bool, "score": float, "threshold": float}` |

### 3.10 Merge / Convert / Upload Events

| event_type | data |
|------------|------|
| `merge_complete` | `{"output_path": "string"}` |
| `convert_complete` | `{"output_path": "string", "format": "string"}` |
| `upload_progress` | `{"bytes_sent": int, "total_bytes": int}` |
| `upload_complete` | `{"repo_url": "string"}` |

### 3.11 EC2 Events

| event_type | data |
|------------|------|
| `ec2_instance_requested` | `{"instance_type": "string"}` |
| `ec2_instance_running` | `{"instance_id": "string", "ip": "string"}` |
| `ec2_instance_terminated` | `{"instance_id": "string", "reason": "string"}` |

---

## 4. Job YAML Format

A single YAML file defines a complete job. Omit any section to skip it.

```yaml
meta:
  name: "string"                  # job display name
  template: "string"              # template this was created from (or 'custom')

execution:
  target: local                   # local | ec2
  ec2:                            # required if target: ec2
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
      output_dir: ~/jobs/job-name/dataset

  - type: data_gen
    config:
      endpoint: "http://localhost:8080/v1"
      model: default
      system_prompt: "You are a helpful assistant."
      user_template: "Answer this question: {{ question }}"
      samples: 1000
      concurrency: 4
      temperature: 0.85
      max_tokens: 512
      quality_threshold: 0.7
      judge_endpoint: "http://localhost:8080/v1"
      judge_model: default
      cross_review: false
      output_dir: ~/jobs/job-name/generated

  - type: finetune
    config:
      mode: standard                # standard | healing
      base_model: /path/to/model
      dataset: ~/jobs/job-name/dataset/train.jsonl
      eval_dataset: ~/jobs/job-name/dataset/eval.jsonl
      output_dir: ~/jobs/job-name/adapter
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
      tokenizer_corpus: ~/jobs/job-name/dataset/train.jsonl
      vocab_size: 32000
      dataset: ~/jobs/job-name/dataset/train.jsonl
      eval_dataset: ~/jobs/job-name/dataset/eval.jsonl
      output_dir: ~/jobs/job-name/model
      training:
        # same fields as finetune.training above

  - type: quant
    config:
      model_path: /path/to/model
      output_dir: ~/jobs/job-name/quants
      output_prefix: my-model
      quant_types:
        - Q4_K_M
        - Q5_K_M
        - TQ3_1S
      imatrix: false
      imatrix_dataset: ~/jobs/job-name/dataset/train.jsonl  # required if imatrix: true

  - type: merge
    config:
      base_model: /path/to/base
      adapter_path: ~/jobs/job-name/adapter
      output_dir: ~/jobs/job-name/merged
      eval_gate:
        enabled: true
        benchmark: perplexity
        threshold: 15.0          # fail if perplexity > threshold
        on_fail: pause           # pause | abort

  - type: prune
    config:
      model_path: /path/to/model
      method: shorthgpt          # llm_pruner | shortgpt | slicegpt | wanda
      pruning_ratio: 0.2         # fraction of parameters to remove
      calibration_dataset: ~/jobs/job-name/dataset/train.jsonl
      output_dir: ~/jobs/job-name/pruned

  - type: eval
    config:
      model_path: /path/to/model
      benchmarks:
        - type: perplexity
          dataset: ~/jobs/job-name/dataset/eval.jsonl
        - type: mmlu
          max_samples: 500
        - type: custom
          dataset: ~/jobs/job-name/dataset/eval.jsonl
          metric: exact_match
      output_dir: ~/jobs/job-name/eval

  - type: convert
    config:
      input_path: /path/to/model
      output_path: ~/jobs/job-name/converted
      format: gguf_f16           # gguf_f16 | gguf_bf16 | safetensors_fp32

  - type: upload
    config:
      source_path: ~/jobs/job-name/quants
      hf_repo: "org/model-name"
      visibility: public         # public | private
      generate_model_card: true
```

---

## 5. Template Format

Templates are YAML files stored in `mad-lab-train/templates/`.  
Pre-built templates: `templates/builtin/`  
Custom templates: `templates/custom/`

```yaml
# templates/builtin/full_finetune.yaml
meta:
  name: full_finetune
  label: "Full Finetune"
  description: "Dataset prep → data generation → QLoRA finetune → GGUF quant"
  is_builtin: true

stages:
  - type: dataset_prep
    defaults:
      train_split: 0.9
      deduplicate: true

  - type: data_gen
    defaults:
      samples: 1000
      concurrency: 4
      temperature: 0.85
      quality_threshold: 0.7

  - type: finetune
    defaults:
      mode: standard
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

  - type: quant
    defaults:
      quant_types: [Q4_K_M]
      imatrix: false
```

**Pre-built templates (6):**

| File | Label | Chain |
|------|-------|-------|
| `gguf_quant.yaml` | GGUF Quant | dataset_prep → quant |
| `full_finetune.yaml` | Full Finetune | dataset_prep → data_gen → finetune → quant |
| `from_scratch.yaml` | From Scratch | dataset_prep → data_gen → pretrain → quant |
| `prune.yaml` | Prune | dataset_prep → data_gen → prune → finetune (healing) |
| `merge.yaml` | Merge | dataset_prep → data_gen → merge → eval → prune → finetune (healing) |
| `eval_standalone.yaml` | Eval Standalone | eval |

---

## 6. Checkpoint Model

```json
{
  "id":           "uuid",
  "job_id":       "uuid",
  "stage_id":     "uuid",
  "sequence":     "int (monotonically increasing per stage)",
  "created_at":   "ISO8601",
  "is_clean":     "bool (false = force-paused, may have incomplete artifacts)",
  "artifact_path": "string (directory containing checkpoint files)",
  "metadata":     "object (stage-specific, see below)"
}
```

**Metadata per stage type:**

| Stage | Metadata fields |
|-------|----------------|
| `finetune` / `pretrain` | `step`, `epoch`, `train_loss`, `eval_loss` |
| `data_gen` | `samples_completed`, `total_samples` |
| `dataset_prep` | `sources_completed`, `total_sources`, `records_written` |
| `quant` | `quant_types_completed`, `current_quant_type` |
| `prune` | `layers_pruned`, `total_layers` |
| `eval` | `benchmarks_completed`, `total_benchmarks` |

---

## 7. Stage Config Schemas

This section documents every field for each stage type with types and defaults.

### 7.1 dataset_prep

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `sources` | `list[SourceConfig]` | — | yes |
| `train_split` | `float` | `0.9` | no |
| `deduplicate` | `bool` | `true` | no |
| `output_dir` | `string` | auto | no |

**SourceConfig types:** `huggingface`, `zim`, `qdrant`, `duckdb`, `raw`

### 7.2 data_gen

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `endpoint` | `string` | — | yes |
| `model` | `string` | `default` | no |
| `system_prompt` | `string` | — | yes |
| `user_template` | `string` | — | yes |
| `samples` | `int` | `1000` | no |
| `concurrency` | `int` | `4` | no |
| `temperature` | `float` | `0.85` | no |
| `max_tokens` | `int` | `512` | no |
| `quality_threshold` | `float 0-1` | `0.7` | no |
| `judge_endpoint` | `string` | same as `endpoint` | no |
| `judge_model` | `string` | same as `model` | no |
| `cross_review` | `bool` | `false` | no |
| `output_dir` | `string` | auto | no |

### 7.3 finetune

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `mode` | `standard \| healing` | `standard` | no |
| `base_model` | `string` | — | yes |
| `dataset` | `string` | auto-wired | no |
| `eval_dataset` | `string` | auto-wired | no |
| `output_dir` | `string` | auto | no |
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

### 7.4 pretrain

Same as `finetune` plus:

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `architecture` | `string (path to arch JSON)` | — | yes |
| `tokenizer_corpus` | `string` | auto-wired | no |
| `vocab_size` | `int` | `32000` | no |

### 7.5 quant

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model_path` | `string` | auto-wired | no |
| `output_dir` | `string` | auto | no |
| `output_prefix` | `string` | job name | no |
| `quant_types` | `list[string]` | `[Q4_K_M]` | no |
| `imatrix` | `bool` | `false` | no |
| `imatrix_dataset` | `string` | auto-wired if imatrix true | no |

### 7.6 merge

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `base_model` | `string` | — | yes |
| `adapter_path` | `string` | auto-wired | no |
| `output_dir` | `string` | auto | no |
| `eval_gate.enabled` | `bool` | `false` | no |
| `eval_gate.benchmark` | `string` | `perplexity` | no |
| `eval_gate.threshold` | `float` | — | if gate enabled |
| `eval_gate.on_fail` | `pause \| abort` | `pause` | no |

### 7.7 prune

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model_path` | `string` | auto-wired | no |
| `method` | `llm_pruner \| shortgpt \| slicegpt \| wanda` | `shortgpt` | no |
| `pruning_ratio` | `float 0-1` | `0.2` | no |
| `calibration_dataset` | `string` | auto-wired | no |
| `output_dir` | `string` | auto | no |

### 7.8 eval

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `model_path` | `string` | auto-wired | no |
| `benchmarks` | `list[BenchmarkConfig]` | — | yes |
| `output_dir` | `string` | auto | no |

**BenchmarkConfig types:** `perplexity`, `mmlu`, `custom`

### 7.9 convert

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `input_path` | `string` | auto-wired | no |
| `output_path` | `string` | auto | no |
| `format` | `gguf_f16 \| gguf_bf16 \| safetensors_fp32` | `gguf_f16` | no |

### 7.10 upload

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `source_path` | `string` | auto-wired | no |
| `hf_repo` | `string` | — | yes |
| `visibility` | `public \| private` | `public` | no |
| `generate_model_card` | `bool` | `true` | no |

---

## Auto-Wiring Rules

When stages are chained, outputs are automatically wired to inputs:

| Producing Stage | Output | Consuming Stage | Auto-wired Field |
|----------------|--------|-----------------|-----------------|
| `dataset_prep` | `output_dir/train.jsonl` | `data_gen` | `input_path` |
| `dataset_prep` | `output_dir/train.jsonl` | `finetune` | `dataset` |
| `dataset_prep` | `output_dir/eval.jsonl` | `finetune` | `eval_dataset` |
| `dataset_prep` | `output_dir/train.jsonl` | `quant` | `imatrix_dataset` |
| `dataset_prep` | `output_dir/train.jsonl` | `prune` | `calibration_dataset` |
| `dataset_prep` | `output_dir/eval.jsonl` | `eval` | benchmark dataset |
| `data_gen` | `output_dir/train.jsonl` | `finetune` | `dataset` (if dataset_prep absent) |
| `finetune` | `output_dir` (adapter) | `merge` | `adapter_path` |
| `pretrain` | `output_dir` | `quant` | `model_path` |
| `merge` | `output_dir` | `prune` | `model_path` |
| `merge` | `output_dir` | `eval` | `model_path` |
| `prune` | `output_dir` | `finetune` (healing) | `base_model` |
| `finetune` | `output_dir/merged` (after merge_and_unload) | `quant` | `model_path` |

If `imatrix: true` in a quant stage and no `dataset_prep` stage is present in the chain, the server returns a validation error at job creation time.

---

*Last updated: 2026-05-02 — MAD-69*
