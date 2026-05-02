"""Worker pool for data generation — manages local and remote llama.cpp servers."""
import asyncio
import subprocess
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class WorkerConfig:
    type: str          # local | ec2
    host: str
    port: int = 8080
    parallel: int = 50
    model: str = ""    # expected model identifier
    # EC2 fields (handled by ec2 bootstrap, not here)
    endpoint: str = field(init=False)

    def __post_init__(self):
        self.endpoint = f"http://{self.host}:{self.port}"


class Worker:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.parallel)
        self.healthy = False
        self._client: httpx.AsyncClient | None = None

    @property
    def endpoint(self) -> str:
        return self.config.endpoint

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.endpoint, timeout=120.0)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = await self._client.get("/health", timeout=5.0)
            self.healthy = resp.status_code == 200
        except Exception:
            self.healthy = False
        return self.healthy

    async def check_model(self, expected_model: str) -> bool:
        """Returns True if the running model matches expected_model."""
        if self._client is None:
            return False
        try:
            resp = await self._client.get("/v1/models", timeout=5.0)
            data = resp.json()
            models = data.get("data", [])
            if not models:
                return False
            running = models[0].get("id", "")
            # Match on model name substring (GGUF filenames are long)
            name = expected_model.split("/")[-1].lower()
            return name in running.lower() or running.lower() in name
        except Exception:
            return False

    async def generate(self, messages: list[dict], temperature: float, max_tokens: int) -> str | None:
        if self._client is None:
            return None
        async with self.semaphore:
            try:
                resp = await self._client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                    timeout=120.0,
                )
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                return None


class WorkerPool:
    """Manages a set of workers with weighted round-robin dispatch."""

    def __init__(self, workers: list[Worker]):
        self.workers = [w for w in workers if w.healthy]
        self._idx = 0

    def pick(self) -> Worker | None:
        if not self.workers:
            return None
        # Weighted round-robin: pick worker with most available semaphore slots
        best = min(self.workers, key=lambda w: w.semaphore._value)
        # Fall back to simple round-robin if all equally loaded
        if best.semaphore._value == self.workers[0].semaphore._value:
            w = self.workers[self._idx % len(self.workers)]
            self._idx += 1
            return w
        return best

    @property
    def total_capacity(self) -> int:
        return sum(w.config.parallel for w in self.workers)


async def prepare_local_worker(cfg_dict: dict, model: str, ctx_size: int) -> Worker:
    """Check-before-kill: verify local llama.cpp server has right model, restart if not."""
    worker = Worker(WorkerConfig(
        type="local",
        host=cfg_dict.get("host", "localhost"),
        port=cfg_dict.get("port", 8080),
        parallel=cfg_dict.get("parallel", 50),
        model=model,
    ))
    await worker.connect()

    alive = await worker.health_check()
    if alive:
        model_ok = await worker.check_model(model)
        if model_ok:
            return worker
        # Wrong model — kill and restart
        _kill_llama_server(worker.config.host, worker.config.port)
        await asyncio.sleep(2)

    # Start fresh llama.cpp server
    started = await _start_llama_server(worker.config.host, worker.config.port, model, ctx_size, cfg_dict.get("parallel", 50))
    worker.healthy = started
    return worker


def _kill_llama_server(host: str, port: int) -> None:
    try:
        if host in ("localhost", "127.0.0.1"):
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True,
            )
            for pid in result.stdout.strip().splitlines():
                subprocess.run(["kill", pid], capture_output=True)
    except Exception:
        pass


async def _start_llama_server(host: str, port: int, model: str, ctx_size: int, parallel: int) -> bool:
    """Start llama-server with the given model. Returns True if healthy within 60s."""
    # For remote hosts, this is a stub — MAD-80 handles SSH-based remote management
    if host not in ("localhost", "127.0.0.1", "mad-lab-main"):
        return False

    cmd = [
        "llama-server",
        "--model", model,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--parallel", str(parallel),
        "--log-disable",
    ]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return False

    # Poll until healthy or timeout
    client = httpx.AsyncClient(base_url=f"http://{host}:{port}", timeout=5.0)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            resp = await client.get("/health")
            if resp.status_code == 200:
                await client.aclose()
                return True
        except Exception:
            pass
        await asyncio.sleep(2)
    await client.aclose()
    return False
