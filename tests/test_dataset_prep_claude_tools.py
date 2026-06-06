"""Integration test: claude_jsonl include_tools path renders tool-interleaved turns.

Drives the async `_parse_claude_session` over a synthetic Claude Code transcript
and asserts (a) include_tools=True yields full tool turns, (b) the default
(flag off) is byte-identical to the prior user+assistant-prose shape.
"""
import asyncio
import json

import pytest

pytest.importorskip("sqlalchemy")  # dataset_prep imports it at module load
from pipeline.executors.dataset_prep import _parse_claude_session


def _write_transcript(tmp_path):
    """A one-turn session: user → assistant(text+tool_use) → tool_result → assistant."""
    lines = [
        {"type": "user", "sessionId": "s1", "cwd": "/x", "gitBranch": "main",
         "timestamp": "t0", "message": {"content": "read the file"}},
        {"type": "assistant", "timestamp": "t1", "message": {
            "model": "claude-x", "content": [
                {"type": "text", "text": "Let me read it."},
                {"type": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "/tmp/f"}},
            ]}},
        {"type": "user", "timestamp": "t2", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "def f(): pass", "is_error": False},
        ]}},
        {"type": "assistant", "timestamp": "t3", "message": {
            "model": "claude-x", "content": [{"type": "text", "text": "Found it."}]}},
    ]
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in lines) + "\n")
    return p


async def _collect(path, include_tools):
    out = []
    async for rec in _parse_claude_session(path, "turn", 0, "claude-code", "organic_work", include_tools):
        out.append(rec)
    return out


def test_include_tools_emits_tool_interleaved_messages(tmp_path):
    path = _write_transcript(tmp_path)
    recs = asyncio.run(_collect(path, include_tools=True))
    assert len(recs) == 1
    msgs = recs[0]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"][0]["function"] == {"name": "Read", "arguments": {"path": "/tmp/f"}}
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "content": "def f(): pass"}
    assert msgs[3]["content"] == "Found it."
    # trace sidecar still populated (memory-trace consumers depend on it)
    assert recs[0]["trace"]["memory_calls"][0]["tool_name"] == "Read"


def test_default_is_backward_compatible(tmp_path):
    path = _write_transcript(tmp_path)
    recs = asyncio.run(_collect(path, include_tools=False))
    assert len(recs) == 1
    msgs = recs[0]["messages"]
    # unchanged shape: exactly user + assistant-prose, no tool turns
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0] == {"role": "user", "content": "read the file"}
    assert msgs[1] == {"role": "assistant", "content": "Let me read it.Found it."}
