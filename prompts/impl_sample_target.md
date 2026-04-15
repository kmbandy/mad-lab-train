# Task: Add sample count target + generation multiplier to mad-lab-train pipeline

## Context

The mad-lab-train pipeline (`npc/pipeline/`) generates synthetic fine-tuning data.
Currently `RunConfig` in `npc/pipeline/schema.py` has:
- `samples_writer: int` — how many samples the writer model generates
- `samples_opus: int` — how many samples the opus model generates
- `hf_target_total: int` — how many HF dataset samples to mix in

There is no concept of a total target or overgeneration-then-filter funnel.
We want to generate e.g. 75-100k raw samples and filter down to ~50k.

## What to implement

### 1. Add to `RunConfig` in `npc/pipeline/schema.py`

```python
target_total: int = 0             # desired post-filter sample count (0 = use samples_writer/opus directly)
generation_multiplier: float = 1.5  # overgenerate by this factor to account for filter losses
```

`target_total` takes precedence over `samples_writer`/`samples_opus` when set > 0.
When `target_total > 0`, per-model sample count should be computed as:
`ceil((target_total * generation_multiplier) / num_generators)`

### 2. Update `npc/pipeline/generate.py`

- At startup, if `run_cfg.target_total > 0`, compute per-model count from the formula above
  and override `samples_writer` / `samples_opus` accordingly
- Add a progress log line showing: "Targeting X raw samples (Yx multiplier) → ~Z post-filter"

### 3. Update `npc/themes/gpu_architecture/run_gpu_architecture.yaml`

Add:
```yaml
target_total: 50000
generation_multiplier: 1.6
```

Remove (or leave as fallback):
```yaml
samples_writer: 400
samples_opus: 200
```

## Files to read first

- `npc/pipeline/schema.py` — RunConfig and all sub-models (understand the full schema)
- `npc/pipeline/generate.py` — how samples_writer/samples_opus are currently consumed
- `npc/pipeline/run.py` — stage_generate, understand how generate.py is invoked
- `npc/themes/gpu_architecture/run_gpu_architecture.yaml` — the config to update

## Constraints

- Do NOT break existing configs that omit `target_total` (default 0 = old behavior)
- Do NOT change validate.py, dataset.py, or any other pipeline stage
- Keep the change minimal — only schema.py, generate.py, and the gpu_architecture run config
- Run `python3 -c "from npc.pipeline.schema import load_run_config"` to verify schema loads cleanly after your change
