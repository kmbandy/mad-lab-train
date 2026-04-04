"""Google Cloud Vertex AI training backend.

Requires:
  - gcloud CLI authenticated: gcloud auth application-default login
  - google-cloud-aiplatform installed: pip install google-cloud-aiplatform
  - finetune.yaml [gcp] section with project, gcs_bucket
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import TrainingBackend


class GCPBackend(TrainingBackend):

    def run(self, train_data: str, eval_data: str) -> None:
        from google.cloud import aiplatform
        gcp_cfg = self.ft_cfg.gcp

        aiplatform.init(project=gcp_cfg.project, location=gcp_cfg.region)

        print(f"[gcp] Uploading training data to gs://{gcp_cfg.gcs_bucket}/...")
        self._upload_data(train_data, eval_data, gcp_cfg.gcs_bucket)

        print(f"[gcp] Launching Vertex AI CustomJob...")
        job = self._launch_job(gcp_cfg)

        print(f"[gcp] Waiting for job '{job.display_name}'...")
        job.wait()  # blocks until complete

        print(f"[gcp] Downloading adapter from GCS...")
        self._download_adapter(gcp_cfg.gcs_bucket)

    def _upload_data(self, train_data: str, eval_data: str, bucket: str) -> None:
        from google.cloud import storage
        client = storage.Client()
        bkt = client.bucket(bucket)
        for local, blob_name in [(train_data, "data/train.jsonl"), (eval_data, "data/eval.jsonl")]:
            if Path(local).exists():
                bkt.blob(blob_name).upload_from_filename(local)
                print(f"  Uploaded {local} → gs://{bucket}/{blob_name}")

    def _launch_job(self, gcp_cfg):
        from google.cloud import aiplatform
        ft = self.ft_cfg
        worker_spec = {
            "machine_spec": {
                "machine_type":      gcp_cfg.machine_type,
                "accelerator_type":  gcp_cfg.accelerator_type,
                "accelerator_count": gcp_cfg.accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": (
                    "us-docker.pkg.dev/vertex-ai/training/"
                    "pytorch-gpu.2-1.py310:latest"
                ),
                "command": ["python3", "train_entrypoint.py"],
                "args": [
                    "--base_model",   ft.base_model,
                    "--num_epochs",   str(ft.num_epochs),
                    "--lora_r",       str(ft.lora_r),
                    "--sequence_len", str(ft.sequence_len),
                    "--output_dir",   f"gs://{gcp_cfg.gcs_bucket}/output/adapter",
                    "--train_data",   f"gs://{gcp_cfg.gcs_bucket}/data/train.jsonl",
                    "--eval_data",    f"gs://{gcp_cfg.gcs_bucket}/data/eval.jsonl",
                ],
            },
        }
        job = aiplatform.CustomJob(
            display_name=f"mad-lab-train-{Path(ft.base_model).name}",
            worker_pool_specs=[worker_spec],
            staging_bucket=f"gs://{gcp_cfg.gcs_bucket}",
        )
        job.run(sync=False)
        return job

    def _download_adapter(self, bucket: str) -> None:
        output_dir = Path(self.ft_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "gsutil", "-m", "cp", "-r",
            f"gs://{bucket}/output/adapter/*",
            str(output_dir) + "/",
        ], check=True)
        print(f"  Adapter saved to {output_dir}")
