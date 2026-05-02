"""Upload/Publish executor — push model artifacts to HuggingFace Hub.

HF token pulled from AWS SSM Parameter Store at /mad-lab/hf-token.
Supports uploading SafeTensors directories, GGUF files, or adapter directories.
Optional model card generated from run metadata.
"""
import os
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor


class UploadExecutor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)

    async def run(self) -> str | None:
        import asyncio
        from pipeline.settings import settings

        cfg = self.config
        hf_repo = cfg["hf_repo"]
        visibility = cfg.get("visibility", "public")
        generate_card = bool(cfg.get("generate_model_card", True))

        run_datasets_dir = (
            Path(os.path.expanduser(settings.log_dir)).parent / "datasets" / str(self.run_id)
        )
        source_path = _resolve_source(cfg, run_datasets_dir)

        await self.emit_event("upload_started", {
            "hf_repo": hf_repo,
            "source_path": str(source_path),
        }, stage_type="upload")

        token = _get_hf_token()

        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(
            None,
            lambda: _upload(
                source_path, hf_repo, visibility, token, generate_card,
                str(self.run_id), self
            ),
        )

        await self.emit_event("upload_complete", {
            "hf_repo": hf_repo,
            "url": url,
        }, stage_type="upload")

        return url

    async def pause(self) -> None:
        pass  # upload is atomic per file — no mid-operation pause

    async def force_pause(self) -> None:
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_hf_token() -> str:
    """Fetch HF token from AWS SSM Parameter Store, fall back to HF_TOKEN env var."""
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    try:
        import boto3
        ssm = boto3.client("ssm")
        resp = ssm.get_parameter(Name="/mad-lab/hf-token", WithDecryption=True)
        return resp["Parameter"]["Value"]
    except Exception as e:
        raise RuntimeError(
            f"Cannot retrieve HF token from SSM /mad-lab/hf-token and HF_TOKEN env not set: {e}"
        )


def _resolve_source(cfg: dict, run_datasets_dir: Path) -> Path:
    if cfg.get("source_path"):
        return Path(os.path.expanduser(cfg["source_path"]))
    # Prefer quant output (GGUFs), then merge, finetune, pretrain
    for subdir in ("quant", "merge", "finetune", "pretrain"):
        candidate = run_datasets_dir / subdir
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "upload.source_path not set and no upstream executor output found"
    )


def _upload(
    source_path: Path,
    hf_repo: str,
    visibility: str,
    token: str,
    generate_card: bool,
    run_id: str,
    executor,
) -> str:
    """Create/update HF repo and upload files. Returns repo URL."""
    import asyncio
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=hf_repo,
        repo_type="model",
        private=(visibility == "private"),
        exist_ok=True,
    )

    if generate_card:
        card = _generate_model_card(hf_repo, run_id)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=hf_repo,
            repo_type="model",
            token=token,
        )

    if source_path.is_file():
        # Single file (e.g. one GGUF)
        api.upload_file(
            path_or_fileobj=str(source_path),
            path_in_repo=source_path.name,
            repo_id=hf_repo,
            repo_type="model",
            token=token,
        )
    else:
        # Directory — upload file-by-file for progress events
        files = [f for f in source_path.rglob("*") if f.is_file()]
        for i, fpath in enumerate(files):
            rel = fpath.relative_to(source_path)
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=str(rel),
                repo_id=hf_repo,
                repo_type="model",
                token=token,
            )
            pct = round((i + 1) / len(files) * 100, 1)
            # Emit progress (fire-and-forget from sync context via loop)
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    executor.emit_event("upload_progress", {
                        "file": str(rel),
                        "percent": pct,
                    }, stage_type="upload"),
                    loop,
                )
            except Exception:
                pass

    return f"https://huggingface.co/{hf_repo}"


def _generate_model_card(hf_repo: str, run_id: str) -> str:
    return f"""\
---
license: apache-2.0
tags:
  - mad-lab-train
  - generated
---

# {hf_repo.split("/")[-1]}

Generated by [mad-lab-train](https://github.com/mad-lab-ai/mad-lab-train) — run `{run_id[:8]}`.
"""
