"""AWS SageMaker training backend.

Requires:
  - AWS CLI configured (aws configure or IAM role)
  - boto3 installed: pip install boto3
  - finetune.yaml [aws] section with role_arn and s3_bucket
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .base import TrainingBackend


class AWSBackend(TrainingBackend):

    def run(self, train_data: str, eval_data: str) -> None:
        import boto3
        aws_cfg = self.ft_cfg.aws
        ft      = self.ft_cfg

        print(f"[aws] Uploading training data to s3://{aws_cfg.s3_bucket}/...")
        self._upload_data(train_data, eval_data, aws_cfg.s3_bucket)

        print(f"[aws] Launching SageMaker training job...")
        job_name = self._launch_job(aws_cfg)

        print(f"[aws] Waiting for job '{job_name}'...")
        self._wait_for_job(job_name)

        print(f"[aws] Downloading adapter from S3...")
        self._download_adapter(job_name, aws_cfg.s3_bucket)

    def _upload_data(self, train_data: str, eval_data: str, bucket: str) -> None:
        import boto3
        s3 = boto3.client("s3")
        for local, key in [(train_data, "data/train.jsonl"), (eval_data, "data/eval.jsonl")]:
            if Path(local).exists():
                s3.upload_file(local, bucket, key)
                print(f"  Uploaded {local} → s3://{bucket}/{key}")

    def _launch_job(self, aws_cfg) -> str:
        import boto3
        sm  = boto3.client("sagemaker")
        ft  = self.ft_cfg
        job_name = f"mad-lab-train-{Path(ft.base_model).name}-{int(time.time())}"

        hyperparameters = {
            "base_model":                    ft.base_model,
            "num_epochs":                    str(ft.num_epochs),
            "micro_batch_size":              str(ft.micro_batch_size),
            "gradient_accumulation_steps":   str(ft.gradient_accumulation_steps),
            "learning_rate":                 str(ft.learning_rate),
            "lora_r":                        str(ft.lora_r),
            "lora_alpha":                    str(ft.lora_alpha),
            "lora_dropout":                  str(ft.lora_dropout),
            "sequence_len":                  str(ft.sequence_len),
            "warmup_steps":                  str(ft.warmup_steps),
        }

        image_uri = aws_cfg.image_uri or (
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
            "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
        )

        sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                "TrainingImage":    image_uri,
                "TrainingInputMode": "File",
            },
            RoleArn=aws_cfg.role_arn,
            InputDataConfig=[{
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{aws_cfg.s3_bucket}/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }},
            }],
            OutputDataConfig={"S3OutputPath": f"s3://{aws_cfg.s3_bucket}/output/"},
            ResourceConfig={
                "InstanceType":     aws_cfg.instance_type,
                "InstanceCount":    1,
                "VolumeSizeInGB":   30,
            },
            StoppingCondition={"MaxRuntimeInSeconds": 86400},
            HyperParameters=hyperparameters,
            EnableManagedSpotTraining=aws_cfg.spot,
        )
        return job_name

    def _wait_for_job(self, job_name: str) -> None:
        import boto3
        sm = boto3.client("sagemaker")
        while True:
            response = sm.describe_training_job(TrainingJobName=job_name)
            status = response["TrainingJobStatus"]
            print(f"  [{job_name}] status: {status}")
            if status in ("Completed", "Failed", "Stopped"):
                if status != "Completed":
                    raise RuntimeError(f"SageMaker job failed with status: {status}")
                return
            time.sleep(30)

    def _download_adapter(self, job_name: str, bucket: str) -> None:
        import boto3
        sm = boto3.client("sagemaker")
        response  = sm.describe_training_job(TrainingJobName=job_name)
        model_uri = response["ModelArtifacts"]["S3ModelArtifacts"]
        output_dir = Path(self.ft_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tar_path = output_dir / "model.tar.gz"
        subprocess.run(["aws", "s3", "cp", model_uri, str(tar_path)], check=True)
        subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(output_dir)], check=True)
        tar_path.unlink(missing_ok=True)
        print(f"  Adapter saved to {output_dir}")
