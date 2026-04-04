"""Lambda Labs cloud GPU training backend.

Requires:
  - lambdalabs SDK: pip install lambdalabs
  - SSH key registered with Lambda Labs
  - API key in the env var specified by lambda_labs.api_key_env
  - finetune.yaml [lambda_labs] section with instance_type and ssh_key_name
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .base import TrainingBackend


class LambdaLabsBackend(TrainingBackend):

    def run(self, train_data: str, eval_data: str) -> None:
        import lambdalabs
        ll_cfg  = self.ft_cfg.lambda_labs
        api_key = os.environ.get(ll_cfg.api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"Lambda Labs API key not found — set env var '{ll_cfg.api_key_env}'"
            )

        client   = lambdalabs.Client(api_key=api_key)
        instance = self._launch_instance(client, ll_cfg)
        ip       = instance["ip"]

        try:
            print(f"[lambda] Waiting for SSH on {ip}...")
            self._wait_for_ssh(ip, ll_cfg.ssh_key_name)

            print(f"[lambda] Uploading training data...")
            self._upload_data(ip, train_data, eval_data, ll_cfg.ssh_key_name)

            print(f"[lambda] Running training...")
            self._run_training(ip, ll_cfg.ssh_key_name)

            print(f"[lambda] Downloading adapter...")
            self._download_adapter(ip, ll_cfg.ssh_key_name)

        finally:
            print(f"[lambda] Terminating instance {instance['id']}...")
            client.terminate_instances([instance["id"]])

    def _launch_instance(self, client, ll_cfg) -> dict:
        response = client.launch_instances(
            region_name="us-west-2",
            instance_type_name=ll_cfg.instance_type,
            ssh_key_names=[ll_cfg.ssh_key_name],
            quantity=1,
        )
        instance_id = response["data"]["instance_ids"][0]
        # Poll until running
        for _ in range(60):
            info = client.get_instance(instance_id)
            if info.get("status") == "active":
                return info
            time.sleep(10)
        raise TimeoutError("Lambda instance never became active")

    def _wait_for_ssh(self, ip: str, ssh_key: str, timeout: int = 300) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 "-i", f"~/.ssh/{ssh_key}", f"ubuntu@{ip}", "echo ok"],
                capture_output=True,
            )
            if result.returncode == 0:
                return
            time.sleep(10)
        raise TimeoutError(f"SSH to {ip} timed out after {timeout}s")

    def _upload_data(self, ip: str, train_data: str, eval_data: str, ssh_key: str) -> None:
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", f"~/.ssh/{ssh_key}",
             f"ubuntu@{ip}", "mkdir -p ~/training/data ~/training/output"],
            check=True,
        )
        train_script = Path(__file__).parent.parent / "train.py"
        for local in [train_data, eval_data]:
            if Path(local).exists():
                subprocess.run([
                    "rsync", "-az", "-e", f"ssh -i ~/.ssh/{ssh_key}",
                    local, f"ubuntu@{ip}:~/training/data/",
                ], check=True)
        subprocess.run([
            "rsync", "-az", "-e", f"ssh -i ~/.ssh/{ssh_key}",
            str(train_script), f"ubuntu@{ip}:~/training/",
        ], check=True)

    def _run_training(self, ip: str, ssh_key: str) -> None:
        ft  = self.ft_cfg
        cmd = (
            "cd ~/training && "
            "pip install trl peft bitsandbytes transformers datasets -q && "
            f"python3 train.py "
            f"--base_model '{ft.base_model}' "
            f"--num_epochs {ft.num_epochs} "
            f"--lora_r {ft.lora_r} "
            f"--sequence_len {ft.sequence_len} "
            f"--output_dir ~/training/output/adapter "
            f"--train_data ~/training/data/train.jsonl "
            f"--eval_data ~/training/data/eval.jsonl"
        )
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-i", f"~/.ssh/{ssh_key}", f"ubuntu@{ip}", cmd],
            check=True,
        )

    def _download_adapter(self, ip: str, ssh_key: str) -> None:
        output_dir = Path(self.ft_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-az", "-e", f"ssh -i ~/.ssh/{ssh_key}",
            f"ubuntu@{ip}:~/training/output/adapter/",
            str(output_dir) + "/",
        ], check=True)
        print(f"  Adapter saved to {output_dir}")
