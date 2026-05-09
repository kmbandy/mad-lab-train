"""EC2 spot instance executor.

Provisions a spot instance, bootstraps it, runs a job command remotely,
streams logs back as pipeline events, and terminates the instance on any outcome.

ec2_config keys (from Run.ec2_config, may be overridden per-stage config):
  instance_type          str     e.g. "i3en.6xlarge"
  ami_id                 str     Ubuntu 22.04 AMI for the target region
  region                 str     default "us-east-1"
  max_spot_price         str     e.g. "0.50" ($/hr ceiling)
  iam_instance_profile   str     profile name; must have SSM + S3 + EC2 policies
  subnet_id              str     optional
  security_group_ids     list    optional
  s3_bucket              str     bucket for rsync'd artifacts/output
  volume_size_gb         int     default 50
  job_cmd                str     shell command to run on the instance
  bootstrap_cmds         list    extra shell lines appended to base bootstrap
  spot_warn_threshold    float   emit warning event if current price > this ($/hr)
  use_ssm                bool    default True; False falls back to SSH
  key_name               str     EC2 key pair name (only used when use_ssm=False)
  ssh_user               str     default "ubuntu"
  cw_log_group           str     CloudWatch log group; default "/mad-lab/ec2-jobs"

Log streaming strategy (SSM path):
  send_command uses CloudWatchOutputConfig → we tail the log stream with
  get_log_events + nextForwardToken and emit each line as a "log" event.

Log streaming strategy (SSH path):
  subprocess ssh with stdout line-by-line iteration.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.executors.base import BaseExecutor

if TYPE_CHECKING:
    pass

_INSTANCE_READY_TIMEOUT = 300
_SSM_READY_TIMEOUT = 180
_CW_POLL_SEC = 5


class Ec2Executor(BaseExecutor):
    def __init__(self, run_id: uuid.UUID, stage_id: uuid.UUID, config: dict, db: AsyncSession):
        super().__init__(run_id, stage_id, config, db)
        self._pause_requested = False
        self._force_pause = False
        self._instance_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> str | None:
        self._loop = asyncio.get_event_loop()
        loop = self._loop
        result = await loop.run_in_executor(None, self._run_sync)
        return result

    async def pause(self) -> None:
        self._pause_requested = True

    async def force_pause(self) -> None:
        self._force_pause = True
        if self._instance_id:
            try:
                import boto3
                ec2 = boto3.client("ec2", region_name=self.config.get("region", "us-east-1"))
                ec2.terminate_instances(InstanceIds=[self._instance_id])
            except Exception:
                pass

    # ── Sync execution (runs in thread pool) ─────────────────────────────────

    def _run_sync(self) -> str | None:
        import boto3
        cfg = self.config
        region = cfg.get("region", "us-east-1")
        ec2 = boto3.client("ec2", region_name=region)
        ssm = boto3.client("ssm", region_name=region)

        self._emit("stage_started", {"stage_type": "ec2", "region": region,
                                     "instance_type": cfg.get("instance_type")})

        self._check_spot_price(ec2, cfg)
        if self._force_pause:
            return None

        instance_id = self._launch_spot(ec2, cfg)
        self._instance_id = instance_id
        self._emit("ec2_launched", {"instance_id": instance_id})

        try:
            self._wait_instance_running(ec2, instance_id)
            use_ssm = bool(cfg.get("use_ssm", True))
            host: str | None = None if use_ssm else self._get_public_ip(ec2, instance_id)

            if use_ssm:
                self._wait_ssm_ready(ssm, instance_id)
            else:
                assert host is not None
                self._wait_ssh_ready(host, cfg)

            self._bootstrap(ssm if use_ssm else None, instance_id, cfg, host=host)

            if self._force_pause:
                return None

            job_cmd = cfg.get("job_cmd", "")
            if not job_cmd:
                raise ValueError("ec2_config.job_cmd is required")

            if use_ssm:
                output_path = self._run_ssm(ssm, instance_id, job_cmd, cfg)
            else:
                assert host is not None
                output_path = self._run_ssh(host, job_cmd, cfg)

            if self._pause_requested:
                return None

            return output_path

        finally:
            self._terminate(ec2, instance_id)

    # ── Spot price check ──────────────────────────────────────────────────────

    def _check_spot_price(self, ec2, cfg: dict) -> None:
        threshold = cfg.get("spot_warn_threshold")
        instance_type = cfg.get("instance_type", "")
        if not threshold or not instance_type:
            return
        try:
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=1,
            )
            prices = resp.get("SpotPriceHistory", [])
            if prices:
                current = float(prices[0]["SpotPrice"])
                if current > float(threshold):
                    self._emit("warning", {
                        "message": f"Spot price ${current:.4f}/hr exceeds warn threshold ${threshold}/hr",
                        "current_price": current,
                        "threshold": threshold,
                    })
        except Exception as e:
            self._emit("warning", {"message": f"Spot price check failed: {e}"})

    # ── Instance launch ───────────────────────────────────────────────────────

    def _launch_spot(self, ec2, cfg: dict) -> str:
        instance_type = cfg["instance_type"]
        ami_id = cfg["ami_id"]
        max_price = str(cfg.get("max_spot_price", "0.50"))
        volume_gb = int(cfg.get("volume_size_gb", 50))

        launch_spec: dict = {
            "ImageId": ami_id,
            "InstanceType": instance_type,
            "BlockDeviceMappings": [{
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
        }

        if cfg.get("iam_instance_profile"):
            launch_spec["IamInstanceProfile"] = {"Name": cfg["iam_instance_profile"]}

        if cfg.get("key_name"):
            launch_spec["KeyName"] = cfg["key_name"]

        if cfg.get("security_group_ids"):
            launch_spec["SecurityGroupIds"] = cfg["security_group_ids"]

        if cfg.get("subnet_id"):
            launch_spec["SubnetId"] = cfg["subnet_id"]

        # UserData: enable SSM agent (already present on Ubuntu AMIs)
        launch_spec["UserData"] = (
            "#!/bin/bash\n"
            "snap install amazon-ssm-agent --classic 2>/dev/null || true\n"
            "systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true\n"
            "systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service 2>/dev/null || true\n"
        )

        resp = ec2.request_spot_instances(
            SpotPrice=max_price,
            InstanceCount=1,
            Type="one-time",
            LaunchSpecification=launch_spec,
        )
        request_id = resp["SpotInstanceRequests"][0]["SpotInstanceRequestId"]

        # Wait for spot request to be fulfilled
        deadline = time.monotonic() + _INSTANCE_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._force_pause:
                ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[request_id])
                raise InterruptedError("force_pause: spot request cancelled")
            detail = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[request_id])
            req = detail["SpotInstanceRequests"][0]
            state = req["State"]
            if state == "active" and req.get("InstanceId"):
                return req["InstanceId"]
            if state in ("cancelled", "closed", "failed"):
                raise RuntimeError(f"Spot request {request_id} ended with state: {state}")
            time.sleep(10)

        raise TimeoutError(f"Spot instance not fulfilled within {_INSTANCE_READY_TIMEOUT}s")

    # ── Wait helpers ──────────────────────────────────────────────────────────

    def _wait_instance_running(self, ec2, instance_id: str) -> None:
        self._emit("log", {"line": f"Waiting for {instance_id} to enter running state..."})
        deadline = time.monotonic() + _INSTANCE_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._force_pause:
                return
            resp = ec2.describe_instances(InstanceIds=[instance_id])
            state = resp["Reservations"][0]["Instances"][0]["State"]["Name"]
            if state == "running":
                return
            if state in ("terminated", "shutting-down"):
                raise RuntimeError(f"Instance {instance_id} entered unexpected state: {state}")
            time.sleep(10)
        raise TimeoutError(f"Instance {instance_id} not running within {_INSTANCE_READY_TIMEOUT}s")

    def _wait_ssm_ready(self, ssm, instance_id: str) -> None:
        self._emit("log", {"line": f"Waiting for SSM agent on {instance_id}..."})
        deadline = time.monotonic() + _SSM_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._force_pause:
                return
            resp = ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [instance_id]}]
            )
            if resp["InstanceInformationList"]:
                ping = resp["InstanceInformationList"][0].get("PingStatus", "")
                if ping == "Online":
                    return
            time.sleep(10)
        raise TimeoutError(f"SSM agent not online for {instance_id} within {_SSM_READY_TIMEOUT}s")

    def _wait_ssh_ready(self, host: str, _cfg: dict = {}) -> None:
        import socket
        self._emit("log", {"line": f"Waiting for SSH on {host}..."})
        deadline = time.monotonic() + _SSM_READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._force_pause:
                return
            try:
                with socket.create_connection((host, 22), timeout=5):
                    return
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(10)
        raise TimeoutError(f"SSH not reachable on {host} within {_SSM_READY_TIMEOUT}s")

    def _get_public_ip(self, ec2, instance_id: str) -> str:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        return resp["Reservations"][0]["Instances"][0]["PublicIpAddress"]

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def _bootstrap(self, ssm, instance_id: str, cfg: dict, host: str | None = None) -> None:
        self._emit("log", {"line": "Running bootstrap..."})
        base = [
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update -qq",
            "apt-get install -y -qq git python3-pip python3-venv awscli unzip",
        ]
        extra = cfg.get("bootstrap_cmds", [])
        cmds = "\n".join(base + extra)

        if ssm is not None:
            self._ssm_run_wait(ssm, instance_id, cmds, timeout=600)
        else:
            assert host is not None
            self._ssh_run(host, cmds, cfg, timeout=600)

    # ── SSM job execution with CW log streaming ───────────────────────────────

    def _run_ssm(self, ssm, instance_id: str, job_cmd: str, cfg: dict) -> str | None:
        import boto3
        cw = boto3.client("logs", region_name=cfg.get("region", "us-east-1"))
        log_group = cfg.get("cw_log_group", "/mad-lab/ec2-jobs")

        self._ensure_log_group(cw, log_group)

        resp = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [job_cmd]},
            CloudWatchOutputConfig={
                "CloudWatchLogGroupName": log_group,
                "CloudWatchOutputEnabled": True,
            },
            TimeoutSeconds=86400,
        )
        command_id = resp["Command"]["CommandId"]
        log_stream = f"{command_id}/{instance_id}/aws-runShellScript/stdout"

        self._emit("log", {"line": f"Job started — SSM command {command_id}"})
        self._tail_cw_stream(cw, log_group, log_stream, ssm, command_id, instance_id, cfg)

        inv = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        exit_code = inv.get("ResponseCode", -1)
        if exit_code != 0:
            raise RuntimeError(f"Remote job exited with code {exit_code}: {inv.get('StandardErrorContent', '')[:500]}")

        return cfg.get("s3_output_path")

    def _ensure_log_group(self, cw, log_group: str) -> None:
        try:
            cw.create_log_group(logGroupName=log_group)
        except cw.exceptions.ResourceAlreadyExistsException:
            pass

    def _tail_cw_stream(self, cw, log_group: str, log_stream: str,
                        ssm, command_id: str, instance_id: str, _cfg: dict) -> None:
        next_token: str | None = None
        stream_ready = False
        deadline = time.monotonic() + 60

        # Wait for log stream to appear
        while not stream_ready and time.monotonic() < deadline:
            try:
                cw.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
                stream_ready = True
            except cw.exceptions.ResourceAlreadyExistsException:
                stream_ready = True
            except Exception:
                time.sleep(3)

        while True:
            if self._force_pause:
                ssm.cancel_command(CommandId=command_id, InstanceIds=[instance_id])
                break

            # Check job status
            try:
                inv = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
                status = inv.get("StatusDetails", "")
                done = status in ("Success", "Failed", "Cancelled", "TimedOut",
                                  "DeliveryTimedOut", "ExecutionTimedOut")
            except Exception:
                done = False
                status = ""

            # Drain log lines
            try:
                kw = {"logGroupName": log_group, "logStreamName": log_stream,
                      "startFromHead": True}
                if next_token:
                    kw["nextToken"] = next_token
                resp = cw.get_log_events(**kw)
                for event in resp.get("events", []):
                    line = event.get("message", "").rstrip()
                    if line:
                        self._emit("log", {"line": line})
                next_token = resp.get("nextForwardToken")
            except Exception:
                pass

            if done:
                break

            if self._pause_requested:
                ssm.cancel_command(CommandId=command_id, InstanceIds=[instance_id])
                break

            time.sleep(_CW_POLL_SEC)

    def _ssm_run_wait(self, ssm, instance_id: str, cmd: str, timeout: int = 300) -> None:
        resp = ssm.send_command(
            InstanceIds=[instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [cmd]},
            TimeoutSeconds=timeout,
        )
        command_id = resp["Command"]["CommandId"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            inv = ssm.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
            status = inv.get("StatusDetails", "")
            if status in ("Success", "Failed", "Cancelled", "TimedOut"):
                if status != "Success":
                    raise RuntimeError(f"SSM command failed ({status}): {inv.get('StandardErrorContent', '')[:300]}")
                return
            time.sleep(5)
        raise TimeoutError(f"SSM command did not complete within {timeout}s")

    # ── SSH job execution ─────────────────────────────────────────────────────

    def _run_ssh(self, host: str, job_cmd: str, cfg: dict, timeout: int = 86400) -> str | None:  # noqa: ARG002
        import subprocess
        user = cfg.get("ssh_user", "ubuntu")
        key_path = cfg.get("key_path", "")

        ssh_args = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout=30",
        ]
        if key_path:
            ssh_args += ["-i", key_path]
        ssh_args += [f"{user}@{host}", job_cmd]

        proc = subprocess.Popen(ssh_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if self._force_pause or self._pause_requested:
                proc.terminate()
                break
            stripped = line.rstrip()
            if stripped:
                self._emit("log", {"line": stripped})

        proc.wait()
        if proc.returncode != 0 and not self._force_pause and not self._pause_requested:
            raise RuntimeError(f"SSH job exited with code {proc.returncode}")

        return cfg.get("s3_output_path")

    def _ssh_run(self, host: str, cmd: str, cfg: dict, timeout: int = 300) -> None:
        import subprocess
        user = cfg.get("ssh_user", "ubuntu")
        key_path = cfg.get("key_path", "")
        ssh_args = [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout=30",
        ]
        if key_path:
            ssh_args += ["-i", key_path]
        ssh_args += [f"{user}@{host}", cmd]
        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"SSH command failed: {result.stderr[:300]}")

    # ── Teardown ──────────────────────────────────────────────────────────────

    def _terminate(self, ec2, instance_id: str) -> None:
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
            self._emit("ec2_terminated", {"instance_id": instance_id})
        except Exception as e:
            self._emit("warning", {"message": f"Failed to terminate {instance_id}: {e}"})

    # ── Emit helper (thread-safe) ─────────────────────────────────────────────

    def _emit(self, event_type: str, data: dict) -> None:
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self.emit_event(event_type, data, stage_type="ec2"),
            self._loop,
        )
