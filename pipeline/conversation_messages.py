"""Render an ordered conversation event stream into canonical chat messages.

Source-agnostic and pure (stdlib only). Any producer — Claude Code transcripts,
agent logs, synthetic tool-use data — emits an ordered list of events; this turns
them into the OpenAI-style messages that ``tokenizer.apply_chat_template``
consumes, including tool turns (assistant ``tool_calls`` + ``role: "tool"``).

Reused across the training pipeline: SFT datasets, ml8 calibration corpora, and
future dataset creation. Keeping it a pure function (events in, messages out)
means the messy, source-specific parsing lives in each source iterator while the
canonical rendering lives here, tested once.

Event shapes (ordered list of dicts; ``kind`` selects the variant):
    {"kind": "user",        "text": str}
    {"kind": "assistant",   "text": str}                      # assistant prose
    {"kind": "tool_use",    "id": str, "name": str, "input": dict}
    {"kind": "tool_result", "tool_use_id": str,
                            "content": str | list | dict, "is_error": bool}

Message shapes (OpenAI canonical):
    {"role": "user",      "content": str}
    {"role": "assistant", "content": str,                     # tool_calls omitted if none
                          "tool_calls": [
                              {"id": str, "type": "function",
                               "function": {"name": str, "arguments": dict}}]}
    {"role": "tool", "tool_call_id": str, "content": str}

``arguments`` is kept as the native dict; ``apply_chat_template`` serializes it
per the target model's template (Qwen/Hermes-style ``| tojson`` etc.).
"""
from __future__ import annotations


def tool_result_to_text(content) -> str:
    """Flatten a tool_result ``content`` (str | list-of-blocks | dict | None) to text.

    Tool results arrive in several shapes: a bare string, a list of
    ``{"type": "text", "text": ...}`` blocks (or bare strings), or a dict. Join
    all text fragments with newlines; unknown shapes contribute nothing.
    """
    return "\n".join(_iter_result_text(content))


def _iter_result_text(content):
    if content is None:
        return
    if isinstance(content, str):
        if content:
            yield content
        return
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if item:
                    yield item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    if text:
                        yield text
                elif isinstance(text, list):
                    yield from _iter_result_text(text)
        return
    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or ""
        if isinstance(text, str):
            if text:
                yield text
        elif isinstance(text, list):
            yield from _iter_result_text(text)


def events_to_messages(events: list[dict]) -> list[dict]:
    """Ordered conversation events → canonical OpenAI chat messages with tool turns.

    Consecutive assistant text + tool_use events collapse into ONE assistant
    message (content + tool_calls). A tool_result flushes the pending assistant
    message (so the turn that issued the calls precedes its results) and emits a
    ``role: "tool"`` message. User events flush and emit a user message.
    """
    out: list[dict] = []
    # Pending assistant message accumulated across consecutive assistant/tool_use
    # events; flushed on a user event, a tool_result, or end-of-stream.
    asst_text: list[str] = []
    asst_calls: list[dict] = []

    def flush_assistant() -> None:
        nonlocal asst_text, asst_calls
        if not asst_text and not asst_calls:
            return
        msg: dict = {"role": "assistant", "content": "".join(asst_text)}
        if asst_calls:
            msg["tool_calls"] = asst_calls
        out.append(msg)
        asst_text = []
        asst_calls = []

    for ev in events:
        kind = ev.get("kind")
        if kind == "user":
            flush_assistant()
            out.append({"role": "user", "content": ev.get("text", "")})
        elif kind == "assistant":
            asst_text.append(ev.get("text", ""))
        elif kind == "tool_use":
            asst_calls.append({
                "id": ev.get("id", ""),
                "type": "function",
                "function": {
                    "name": ev.get("name", ""),
                    "arguments": ev.get("input", {}) or {},
                },
            })
        elif kind == "tool_result":
            flush_assistant()
            out.append({
                "role": "tool",
                "tool_call_id": ev.get("tool_use_id", ""),
                "content": tool_result_to_text(ev.get("content")),
            })
        # unknown kinds are ignored

    flush_assistant()
    return out
