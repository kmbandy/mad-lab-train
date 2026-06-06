"""Tests for pipeline.conversation_messages — ordered events → tool-aware messages.

Pure-function component: no I/O, no async, no deps. Seeds the test convention
for mad-lab-train (run: `python -m pytest tests/ -q` from repo root).
"""
from pipeline.conversation_messages import events_to_messages, tool_result_to_text


# ── tool_result_to_text (content normalization) ──────────────────

def test_tool_result_plain_string():
    assert tool_result_to_text("hello") == "hello"


def test_tool_result_list_of_text_blocks():
    content = [{"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]
    assert tool_result_to_text(content) == "line1\nline2"


def test_tool_result_list_of_bare_strings():
    assert tool_result_to_text(["a", "b"]) == "a\nb"


def test_tool_result_none_is_empty():
    assert tool_result_to_text(None) == ""


# ── events_to_messages ───────────────────────────────────────────

def test_simple_user_assistant():
    events = [
        {"kind": "user", "text": "hi"},
        {"kind": "assistant", "text": "hello there"},
    ]
    assert events_to_messages(events) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


def test_tool_use_and_result_interleaved():
    events = [
        {"kind": "user", "text": "read the file"},
        {"kind": "assistant", "text": "Let me read it."},
        {"kind": "tool_use", "id": "call_1", "name": "Read", "input": {"path": "/tmp/f"}},
        {"kind": "tool_result", "tool_use_id": "call_1", "content": "def f(): pass", "is_error": False},
        {"kind": "assistant", "text": "Found it."},
    ]
    msgs = events_to_messages(events)
    assert msgs[0] == {"role": "user", "content": "read the file"}
    # assistant that made the call: text + tool_calls in one message
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Let me read it."
    assert msgs[1]["tool_calls"] == [
        {"id": "call_1", "type": "function",
         "function": {"name": "Read", "arguments": {"path": "/tmp/f"}}},
    ]
    # tool result turn
    assert msgs[2] == {"role": "tool", "tool_call_id": "call_1", "content": "def f(): pass"}
    # assistant continuation after the result
    assert msgs[3] == {"role": "assistant", "content": "Found it."}


def test_multiple_tool_uses_one_assistant_turn():
    events = [
        {"kind": "user", "text": "do two things"},
        {"kind": "tool_use", "id": "a", "name": "X", "input": {}},
        {"kind": "tool_use", "id": "b", "name": "Y", "input": {"k": 1}},
        {"kind": "tool_result", "tool_use_id": "a", "content": "ra"},
        {"kind": "tool_result", "tool_use_id": "b", "content": "rb"},
    ]
    msgs = events_to_messages(events)
    # one assistant message holding both calls (no text), then two tool turns
    asst = msgs[1]
    assert asst["role"] == "assistant"
    assert [tc["id"] for tc in asst["tool_calls"]] == ["a", "b"]
    assert msgs[2] == {"role": "tool", "tool_call_id": "a", "content": "ra"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "b", "content": "rb"}


def test_tool_result_block_content_is_flattened():
    events = [
        {"kind": "user", "text": "q"},
        {"kind": "tool_use", "id": "1", "name": "Grep", "input": {}},
        {"kind": "tool_result", "tool_use_id": "1",
         "content": [{"type": "text", "text": "match1"}, {"type": "text", "text": "match2"}]},
    ]
    msgs = events_to_messages(events)
    assert msgs[-1] == {"role": "tool", "tool_call_id": "1", "content": "match1\nmatch2"}


def test_empty_events():
    assert events_to_messages([]) == []


def test_assistant_text_with_no_tool_calls_has_no_tool_calls_key():
    msgs = events_to_messages([{"kind": "user", "text": "u"}, {"kind": "assistant", "text": "a"}])
    assert "tool_calls" not in msgs[1]
