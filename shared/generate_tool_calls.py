#!/usr/bin/env python3
"""Generate synthetic tool-calling training data for eng-2 and the 1B model.

Produces ShareGPT-format JSONL where the assistant's tool invocations use the
Qwen3 <tool_call> format (first priority in nanobot's extractor):

    <tool_call>{"name": "memory_search", "arguments": {"query": "..."}}</tool_call>

Tool results come back as a "tool" role turn (ChatML standard).

Output: /home/kmbandy/mad-lab-mcp/datasets/tool_calls.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx

OUT_PATH = Path("/home/kmbandy/mad-lab-mcp/datasets/tool_calls.jsonl")
SEED = 42
random.seed(SEED)

# Use eng-1 (localhost:8080) — falls back to arch-1 on main PC if eng-1 is down.
# Both expose an OpenAI-compatible /v1/chat/completions endpoint.
_ENDPOINTS = [
    "http://192.168.1.15:8080/v1/chat/completions",    # arch-1 (GPT-OSS-20B, main PC)
    "http://127.0.0.1:8080/v1/chat/completions",       # eng-1 (omnicoder, GTX 1070) fallback
]

def _active_endpoint() -> str:
    for url in _ENDPOINTS:
        try:
            r = httpx.get(url.replace("/chat/completions", "/models"), timeout=3)
            if r.status_code < 500:
                print(f"  Using endpoint: {url}")
                return url
        except Exception:
            pass
    raise RuntimeError("No local llama-server reachable. Start eng-1 or arch-1 first.")

# ---------------------------------------------------------------------------
# Tool schemas (subset of what eng-2 actually has)
# ---------------------------------------------------------------------------

TOOLS_DOC = """Available tools:

memory_search(query: str, limit: int = 5) -> list[dict]
  Search ChromaDB memory for relevant past context.
  Returns: list of {content, type, source, created_at}

memory_write(content: str, agent: str, type: str, source: str) -> dict
  Store a fact or decision in long-term memory.
  Returns: {id, status}

task_get_pending() -> list[dict]
  List all pending tasks assigned to this agent.
  Returns: list of {id, title, description, assignee, status, created_at}

task_claim(task_id: str) -> dict
  Claim a pending task so others know you are working on it.
  Returns: {id, status, claimed_by}

task_complete(task_id: str, result: str) -> dict
  Mark a task as complete and record the result.
  Returns: {id, status, result}

github_list_issues(owner: str, repo: str, state: str = "open") -> list[dict]
  List GitHub issues for a repository.
  Returns: list of {number, title, body, labels, state}

github_issue_read(owner: str, repo: str, issue_number: int) -> dict
  Read full details of a GitHub issue including comments.
  Returns: {number, title, body, comments: list[{author, body}]}

github_get_file_contents(owner: str, repo: str, path: str) -> dict
  Read a file from a GitHub repository.
  Returns: {content, encoding, sha}

github_search_code(query: str, owner: str = None, repo: str = None) -> list[dict]
  Search for code across GitHub repos.
  Returns: list of {path, repo, snippet}

web_search(query: str) -> list[dict]
  Search the web for information.
  Returns: list of {title, url, snippet}

exec_code(code: str, language: str = "python") -> dict
  Execute a code snippet and return output.
  Returns: {stdout, stderr, exit_code}
"""

SYSTEM_PROMPT = (
    "You are mad-lab-nanobot-eng-2, an expert software engineering assistant "
    "in the mad-lab AI fleet. You help with code review, debugging, implementing features, "
    "and managing engineering tasks. You have access to memory, task management, "
    "GitHub integration, web search, and code execution tools. "
    "Use tools proactively when they would help you give a better answer. "
    "Be concise and technical. "
    f"\n\n{TOOLS_DOC}"
)

# ---------------------------------------------------------------------------
# Scenario templates
# ---------------------------------------------------------------------------

SCENARIOS = [
    # ── Memory search ────────────────────────────────────────────────────────
    {
        "category": "memory_search",
        "user_templates": [
            "What do we know about the {topic} setup?",
            "Have we had any issues with {topic} before?",
            "What's the context on {topic}?",
            "Remind me what we decided about {topic}.",
            "What was the outcome of the {topic} work?",
        ],
        "topics": [
            "ChromaDB memory server", "ROCm 6.2.4 workarounds", "bitsandbytes CUDA",
            "nanobot tool calling", "LoRA fine-tuning", "Kaggle GPU kernels",
            "TimescaleDB quant stack", "Discord bot integration", "MCP server setup",
            "Ministral-8B deployment",
        ],
        "tool": "memory_search",
        "tool_result_template": '[{{"content": "Past context about {topic}: {detail}", "type": "project", "source": "assistant", "created_at": "2026-03-{day:02d}"}}]',
        "details": [
            "configuration notes from last session", "issue resolved by setting env var",
            "deployed successfully on port 18792", "uses streamableHttp transport",
            "requires pure transformer models only", "QLoRA r=16 alpha=32",
        ],
    },

    # ── Memory write ─────────────────────────────────────────────────────────
    {
        "category": "memory_write",
        "user_templates": [
            "Remember that {fact}.",
            "Store the decision: {fact}.",
            "Log this: {fact}.",
            "Make a note that {fact}.",
        ],
        "facts": [
            "Ministral-8B fine-tune completed with eval_loss=0.62",
            "BNB_CUDA_VERSION must be set before bitsandbytes import on Kaggle",
            "nanobot-eng-2 now uses Ministral-8B-FT as primary model",
            "RX 480 only supports pure transformer models, no SSM/Mamba",
            "TimescaleDB password changed to new value after rotation",
            "GitHub MCP server requires Bearer auth on port 18801",
            "eng-2 fine-tune trained on 15k SO coding samples + 2k tool calling examples",
        ],
        "tool": "memory_write",
        "tool_result_template": '{{"id": "mem-{id}", "status": "written"}}',
    },

    # ── Task pipeline ────────────────────────────────────────────────────────
    {
        "category": "task_pipeline",
        "user_templates": [
            "Check if there are any tasks for you.",
            "Do you have any pending work?",
            "What tasks are in your queue?",
            "Pick up your next task.",
        ],
        "tasks": [
            {"id": "task-abc123", "title": "Add rate limiting to the web scraper", "description": "Hackernews scraper is hitting rate limits. Add exponential backoff with jitter.", "assignee": "nanobot-eng-2"},
            {"id": "task-def456", "title": "Fix memory_search returning stale entries", "description": "Entries older than 30 days showing in search results. Filter by created_at.", "assignee": "nanobot-eng-2"},
            {"id": "task-ghi789", "title": "Add HuggingFace model scraper", "description": "Scrape trending models from huggingface.co/models. Follow same pattern as github_trending.py.", "assignee": "nanobot-eng-2"},
            {"id": "task-jkl012", "title": "Update nanobot config for new model", "description": "Switch eng-2 llama-server to use newly fine-tuned Ministral-8B GGUF.", "assignee": "nanobot-eng-2"},
            {"id": "task-mno345", "title": "Write unit tests for scraper_filter.py", "description": "Add pytest tests covering score ranking and category classification.", "assignee": "nanobot-eng-2"},
        ],
        "tool": "task_get_pending + task_claim",
    },

    # ── Task complete ────────────────────────────────────────────────────────
    {
        "category": "task_complete",
        "user_templates": [
            "Mark task {task_id} as done. {result}",
            "Complete task {task_id}: {result}",
            "I finished {task_id}. {result}",
        ],
        "task_ids": ["task-abc123", "task-def456", "task-ghi789"],
        "results": [
            "Added exponential backoff with max 5 retries. Tested against live endpoint.",
            "Added `created_at__gte` filter using 30-day cutoff. ChromaDB metadata query updated.",
            "Implemented in scrapers/huggingface.py. Returns top 20 trending models with stars and downloads.",
        ],
        "tool": "task_complete",
        "tool_result_template": '{{"id": "{task_id}", "status": "completed", "result": "{result}"}}',
    },

    # ── GitHub issue ─────────────────────────────────────────────────────────
    {
        "category": "github_issue",
        "user_templates": [
            "Look at issue #{number} in {repo}.",
            "What's issue #{number} in {repo} about?",
            "Can you read and then summarize issue #{number} in {repo}?",
            "Read issue #{number} and tell me what needs to be done.",
        ],
        "issues": [
            {"owner": "kmbandy", "repo": "mad-lab-mcp", "number": 3,
             "title": "Fine-tune Ministral-8B on SO/GH coding dataset",
             "body": "Base model: mistralai/Ministral-8B-Instruct-2512\nTraining: QLoRA 4-bit r=16 ChatML seq_len 2048\nData: SO ZIM extraction + tool calling examples\nTarget: outperform vanilla Ministral on HumanEval"},
            {"owner": "kmbandy", "repo": "nanobot", "number": 20,
             "title": "MCP auto-reconnect on connection drop",
             "body": "MCP server connections occasionally drop after idle periods. Add reconnect logic with exponential backoff in mcp.py."},
            {"owner": "kmbandy", "repo": "mad-lab-scripts", "number": 31,
             "title": "Strategy bot scaffold",
             "body": "Build the strategy bot orchestrator.py + system_prompt.txt. Model: Nemotron-4B on llama-server. DB schema: strategy_trades/positions/risk_decisions tables."},
        ],
        "tool": "github_issue_read",
        "tool_result_template": None,  # built dynamically
    },

    # ── GitHub file read ──────────────────────────────────────────────────────
    {
        "category": "github_file",
        "user_templates": [
            "Read {path} from {repo}.",
            "Show me {path} in {repo}.",
            "What does {path} look like in {repo}?",
        ],
        "files": [
            {"owner": "kmbandy", "repo": "mad-lab-mcp", "path": "scraper_filter.py"},
            {"owner": "kmbandy", "repo": "nanobot", "path": "nanobot/agent/tools/mcp.py"},
            {"owner": "kmbandy", "repo": "mad-lab-scripts", "path": "strategy_bot/context_builder.py"},
        ],
        "tool": "github_get_file_contents",
    },

    # ── Web search ────────────────────────────────────────────────────────────
    {
        "category": "web_search",
        "user_templates": [
            "Search for {query}.",
            "Look up {query}.",
            "What's the latest on {query}?",
            "Can you find docs on {query}?",
        ],
        "queries": [
            "bitsandbytes CUDA 12.8 compatibility fix",
            "Ministral 8B QLoRA training memory usage",
            "ChromaDB vector search performance tuning",
            "nf4 quantization vs fp4 comparison",
            "llama.cpp ROCm gfx803 build flags",
            "peft LoRA r rank selection best practices",
            "HuggingFace transformers 5.x breaking changes",
            "SFTTrainer max_seq_length truncation behavior",
        ],
        "tool": "web_search",
        "result_template": '[{{"title": "{title}", "url": "https://example.com/{slug}", "snippet": "{snippet}"}}]',
    },

    # ── Code execution ────────────────────────────────────────────────────────
    {
        "category": "exec_code",
        "user_templates": [
            "Run this and tell me what happens:\n```python\n{code}\n```",
            "Execute this snippet: {code}",
            "Test this code for me: {code}",
            "Can you run this? {code}",
        ],
        "snippets": [
            "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))",
            "from pathlib import Path; import glob; print(glob.glob('/usr/local/lib/python3.12/dist-packages/bitsandbytes/*.so'))",
            "import json; d={'a': 1, 'b': [1,2,3]}; print(json.dumps(d, indent=2))",
            "x = [i**2 for i in range(10)]; print(sum(x))",
            "import sys; print(sys.version); import torch; print(torch.__version__)",
        ],
        "tool": "exec_code",
        "result_template": '{{"stdout": "{output}", "stderr": "", "exit_code": 0}}',
    },

    # ── Multi-tool: search memory then write ──────────────────────────────────
    {
        "category": "multi_memory",
        "user_templates": [
            "Check if we know anything about {topic}, then store that {fact} is now resolved.",
            "Search memory for {topic} and then log the update: {fact}.",
        ],
        "topics": [
            "bitsandbytes GPU detection", "eng-2 model deployment", "task pipeline performance",
        ],
        "facts": [
            "symlink workaround works for CUDA version mismatch",
            "Ministral-8B fine-tune eval_loss converged at 0.58",
            "task throughput improved after claiming fix",
        ],
        "tool": "memory_search + memory_write",
    },

    # ── Multi-tool: check tasks then work on one ──────────────────────────────
    {
        "category": "multi_task_github",
        "user_templates": [
            "Check your tasks, pick one, and read the relevant GitHub issue for it.",
            "What tasks do you have? Take the first one and look up the GitHub issue.",
        ],
        "tool": "task_get_pending + task_claim + github_issue_read",
    },
]

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _tool_call_block(name: str, arguments: dict) -> str:
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def _generate_conversation(scenario: dict, variation: int) -> dict | None:
    """Ask Claude to generate one complete tool-use conversation for a scenario."""
    cat = scenario["category"]

    # Build a concrete prompt describing the exact scenario variant
    if cat == "memory_search":
        topic = random.choice(scenario["topics"])
        user_msg = random.choice(scenario["user_templates"]).format(topic=topic)
        detail = random.choice(scenario["details"])
        day = random.randint(1, 28)
        tool_result = scenario["tool_result_template"].format(
            topic=topic, detail=detail, day=day)
        meta = {"user": user_msg, "tool": scenario["tool"], "tool_result": tool_result}

    elif cat == "memory_write":
        fact = random.choice(scenario["facts"])
        user_msg = random.choice(scenario["user_templates"]).format(fact=fact)
        uid = random.randint(1000, 9999)
        tool_result = scenario["tool_result_template"].format(id=uid)
        meta = {"user": user_msg, "tool": scenario["tool"], "tool_result": tool_result}

    elif cat == "task_pipeline":
        user_msg = random.choice(scenario["user_templates"])
        task = random.choice(scenario["tasks"])
        pending_result = json.dumps([task])
        claim_result = json.dumps({"id": task["id"], "status": "in_progress", "claimed_by": "nanobot-eng-2"})
        meta = {"user": user_msg, "pending_result": pending_result,
                "claim_result": claim_result, "task": task}

    elif cat == "task_complete":
        idx = variation % len(scenario["task_ids"])
        task_id = scenario["task_ids"][idx]
        result = scenario["results"][idx]
        user_msg = random.choice(scenario["user_templates"]).format(
            task_id=task_id, result=result)
        tool_result = scenario["tool_result_template"].format(
            task_id=task_id, result=result.replace('"', '\\"'))
        meta = {"user": user_msg, "tool": scenario["tool"], "tool_result": tool_result,
                "task_id": task_id, "result": result}

    elif cat == "github_issue":
        issue = random.choice(scenario["issues"])
        user_msg = random.choice(scenario["user_templates"]).format(
            number=issue["number"], repo=issue["repo"])
        tool_result = json.dumps({
            "number": issue["number"],
            "title": issue["title"],
            "body": issue["body"],
            "comments": [],
        })
        meta = {"user": user_msg, "issue": issue, "tool_result": tool_result}

    elif cat == "github_file":
        f = random.choice(scenario["files"])
        user_msg = random.choice(scenario["user_templates"]).format(
            path=f["path"], repo=f["repo"])
        meta = {"user": user_msg, "file": f}

    elif cat == "web_search":
        query = random.choice(scenario["queries"])
        user_msg = random.choice(scenario["user_templates"]).format(query=query)
        slug = query.lower().replace(" ", "-")[:30]
        snippet = f"Comprehensive guide to {query} with examples and troubleshooting steps."
        tool_result = scenario["result_template"].format(
            title=f"{query} - Complete Guide", slug=slug, snippet=snippet)
        meta = {"user": user_msg, "query": query, "tool_result": tool_result}

    elif cat == "exec_code":
        code = random.choice(scenario["snippets"])
        user_msg = random.choice(scenario["user_templates"]).format(code=code)
        output = "True T4\\n" if "cuda" in code else "285\\n" if "sum" in code else "['/path/to/lib.so']\\n"
        tool_result = scenario["result_template"].format(output=output)
        meta = {"user": user_msg, "code": code, "tool_result": tool_result}

    elif cat == "multi_memory":
        topic = random.choice(scenario["topics"])
        fact = random.choice(scenario["facts"])
        user_msg = random.choice(scenario["user_templates"]).format(topic=topic, fact=fact)
        meta = {"user": user_msg, "topic": topic, "fact": fact}

    elif cat == "multi_task_github":
        user_msg = random.choice(scenario["user_templates"])
        meta = {"user": user_msg}

    else:
        return None

    # Ask Claude to write the full conversation
    generator_prompt = f"""You are generating training data for a fine-tuned coding assistant model.

Write a complete multi-turn conversation in ShareGPT JSON format.

SYSTEM PROMPT for the assistant:
{SYSTEM_PROMPT}

SCENARIO:
Category: {cat}
User message: {meta["user"]}
{'Tool name: ' + meta.get("tool", "") if "tool" in meta else ""}
{'Tool result JSON: ' + meta.get("tool_result", "") if "tool_result" in meta else ""}
{json.dumps({k: v for k, v in meta.items() if k not in ("user", "tool", "tool_result")}, indent=2) if len(meta) > 3 else ""}

RULES:
1. Output ONLY valid JSON — a single object with a "conversations" array.
2. Roles: "system", "human", "gpt", "tool"
3. The assistant ("gpt") invokes tools using EXACTLY this format (no other format):
   <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call>
4. Tool results come back as a "tool" role turn with the result JSON as the value.
5. The final "gpt" turn interprets the result and gives a helpful response.
6. For multi-tool scenarios, chain calls: gpt → tool → gpt → tool → gpt (final).
7. Assistant messages before tool calls may include brief reasoning (1 sentence max).
8. Keep responses concise and technical. No filler.
9. For task_pipeline category: first call task_get_pending, then task_claim on the chosen task.
10. The "system" turn is first and contains the system prompt verbatim.

Output only the JSON object, no explanation."""

    try:
        resp = httpx.post(
            _ACTIVE_ENDPOINT,
            json={"model": "default", "max_tokens": 1500,
                  "messages": [{"role": "user", "content": generator_prompt}]},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        # Validate structure
        if not isinstance(data.get("conversations"), list):
            return None
        if len(data["conversations"]) < 3:
            return None
        # Deduplicate consecutive system turns (model sometimes outputs two)
        convos = data["conversations"]
        deduped = [convos[0]]
        for turn in convos[1:]:
            if turn.get("from") == "system" and deduped[-1].get("from") == "system":
                continue
            deduped.append(turn)
        # Ensure system prompt is present
        if deduped[0].get("from") != "system":
            deduped.insert(0, {"from": "system", "value": SYSTEM_PROMPT})
        data["conversations"] = deduped
        data["source"] = "generated_tool_calls"
        data["category"] = cat
        return data
    except Exception as e:
        print(f"    [warn] generation failed for {cat}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tool-calling training data")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of categories to generate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: datasets/tool_calls.jsonl)")
    args = parser.parse_args()

    global _ACTIVE_ENDPOINT
    _ACTIVE_ENDPOINT = _active_endpoint()

    out_path = Path(args.output) if args.output else OUT_PATH
    category_filter = set(args.categories.split(",")) if args.categories else None
    scenarios = [s for s in SCENARIOS if category_filter is None or s["category"] in category_filter]

    n_per_scenario = 500  # ≈ 500 × 10 scenarios = 5000 examples
    total_target = len(scenarios) * n_per_scenario

    print(f"Generating ~{total_target} tool-calling examples → {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: count already-generated examples per category from the output file
    done_per_cat: dict[str, int] = {}
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    cat = d.get("category", "")
                    done_per_cat[cat] = done_per_cat.get(cat, 0) + 1
                except Exception:
                    pass
        total_done = sum(done_per_cat.values())
        if total_done:
            print(f"Resuming — {total_done} examples already written: {done_per_cat}")

    count = sum(done_per_cat.values())
    with out_path.open("a") as f:
        for scenario in scenarios:
            cat = scenario["category"]
            already = done_per_cat.get(cat, 0)
            remaining = n_per_scenario - already
            if remaining <= 0:
                print(f"\n[{cat}] already complete ({already}/{n_per_scenario}), skipping")
                continue
            print(f"\n[{cat}] generating {remaining} more (have {already}/{n_per_scenario})...")
            for i in range(already, n_per_scenario):
                convo = _generate_conversation(scenario, i)
                if convo:
                    f.write(json.dumps(convo) + "\n")
                    f.flush()
                    count += 1
                    print(f"  [{count}] ok", end="\r")
                else:
                    print(f"  [{i}] skip", end="\r")
                time.sleep(0.3)  # gentle rate limiting

    print(f"\nDone. {count} conversations written to {out_path}")


if __name__ == "__main__":
    main()
