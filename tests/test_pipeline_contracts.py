from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pipeline.models import StageType
from pipeline.executors.data_gen import _load_checkpoint, _save_checkpoint
from pipeline.orchestrator import _wire_upstream_artifact
from pipeline.schemas import RunCreate, TemplateResponse


def test_run_requires_inputs_when_chain_cannot_supply_them():
    with pytest.raises(ValidationError, match="base_model is required"):
        RunCreate.model_validate({
            "name": "bad-run",
            "template_name": "custom",
            "stages": [{"stage_type": "finetune", "config": {}}],
        })


def test_run_accepts_compatible_upstream_artifacts():
    run = RunCreate.model_validate({
        "name": "good-run",
        "template_name": "custom",
        "stages": [
            {
                "stage_type": "dataset_prep",
                "config": {"sources": [{"type": "raw", "path": "/tmp/data.jsonl"}]},
            },
            {"stage_type": "finetune", "config": {"base_model": "org/model"}},
            {"stage_type": "merge", "config": {"mode": "adapter"}},
            {"stage_type": "quant", "config": {}},
        ],
    })
    assert len(run.stages) == 4


def test_scheduled_time_requires_timezone():
    with pytest.raises(ValidationError, match="must include a timezone"):
        RunCreate.model_validate({
            "name": "scheduled",
            "template_name": "custom",
            "scheduled_for": "2026-07-15T12:00:00",
            "stages": [{"stage_type": "eval", "config": {"model_path": "/tmp/model"}}],
        })


def test_template_defaults_are_normalized_to_config():
    template = TemplateResponse.model_validate({
        "id": "74d76b37-6b81-46fe-b560-fd276fcca992",
        "name": "quant",
        "label": "Quant",
        "description": "",
        "chain": [{"stage_type": "quant", "defaults": {"quant_types": ["Q4_K_M"]}}],
        "is_builtin": True,
        "created_at": "2026-07-15T12:00:00Z",
    })
    assert template.chain[0]["config"]["quant_types"] == ["Q4_K_M"]
    assert "defaults" not in template.chain[0]


def test_upstream_output_is_injected_without_overriding_explicit_path():
    previous = SimpleNamespace(
        stage_type=StageType.finetune,
        output_path="/runs/one/finetune",
        config=SimpleNamespace(config={"base_model": "org/base"}),
    )
    stage = SimpleNamespace(stage_type=StageType.merge)
    config = {"mode": "adapter"}

    _wire_upstream_artifact(stage, previous, config)

    assert config["adapter_path"] == "/runs/one/finetune"
    assert config["base_model"] == "org/base"

    explicit = {"model_path": "/models/chosen"}
    _wire_upstream_artifact(SimpleNamespace(stage_type=StageType.quant), previous, explicit)
    assert explicit["model_path"] == "/models/chosen"


def test_checkpoint_snapshots_can_be_selected(tmp_path):
    first = _save_checkpoint(tmp_path, 100)
    second = _save_checkpoint(tmp_path, 200)

    assert _load_checkpoint(tmp_path, str(first))["samples_done"] == 100
    assert _load_checkpoint(tmp_path, str(second))["samples_done"] == 200
    assert _load_checkpoint(tmp_path)["samples_done"] == 200


def test_data_generation_checkpoint_records_output_boundary(tmp_path):
    checkpoint = _save_checkpoint(tmp_path, 100, output_bytes=4096)
    data = _load_checkpoint(tmp_path, str(checkpoint))
    assert data == {"samples_done": 100, "output_bytes": 4096}
