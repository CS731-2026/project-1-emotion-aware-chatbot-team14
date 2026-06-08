"""Claude Code CLI as an LLM provider.

Spawns the `claude` CLI in non-interactive mode and returns the model's reply.
No API key, auth is handled by the user's local Claude Code login (Max/Pro
subscription via `claude login`). Useful for local development to avoid paid
API costs.

Conversation history is serialized into stdin as plain-text role markers; the
system prompt is passed via the --system-prompt flag.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from .base import LLMProvider, Message

logger = logging.getLogger(__name__)


def _serialize_history(messages: list[Message]) -> str:
    """Render non-system messages as a conversation transcript for stdin."""
    lines: list[str] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        lines.append(f"[{role}]: {m['content']}")
    return "\n\n".join(lines)


def _collect_system_prompt(messages: list[Message]) -> str:
    """Concatenate all system messages in order (joined with blank lines)."""
    return "\n\n".join(m["content"] for m in messages if m["role"] == "system")


class ClaudeCodeProvider(LLMProvider):
    """LLM provider backed by the local `claude` CLI.

    Each call spawns a fresh subprocess, no session persistence. The CLI's
    JSON envelope is parsed to extract the model's reply text.
    """

    def __init__(
        self,
        model: str = "haiku",
        api_key: str | None = None,
        cli_path: str | None = None,
        timeout_sec: int | None = None,
    ) -> None:
        # api_key is accepted (and ignored) so the factory can pass it
        # uniformly across providers. Claude Code auths via the local CLI.
        _ = api_key
        self._model = model
        self._cli = cli_path or os.getenv("CLAUDE_CLI", "claude")
        self._timeout = timeout_sec or int(os.getenv("CLAUDE_CLI_TIMEOUT_SEC", "180"))

    def chat(self, messages: list[Message]) -> str:
        system_prompt = _collect_system_prompt(messages)
        transcript = _serialize_history(messages)

        cmd = [
            self._cli,
            "--print",
            "--output-format", "json",
            "--model", self._model,
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        try:
            completed = subprocess.run(
                cmd,
                input=transcript,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"claude CLI timed out after {self._timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"claude CLI not found at '{self._cli}'. "
                "Install Claude Code (https://docs.claude.com/en/docs/claude-code) "
                "or set CLAUDE_CLI to its path."
            ) from e

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or "(no stderr)"
            raise RuntimeError(
                f"claude CLI exited with code {completed.returncode}: {stderr}"
            )

        # Envelope shape: {"type": "result", "result": "<text>", "is_error": bool, ...}
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"claude CLI returned non-JSON output: {completed.stdout[:200]!r}"
            ) from e

        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude CLI reported error: {envelope.get('result') or envelope}"
            )

        result = envelope.get("result")
        if not isinstance(result, str):
            raise RuntimeError(
                f"claude CLI envelope missing 'result' string: {envelope}"
            )
        return result

    @property
    def provider_name(self) -> str:
        return "claude-code"

    @property
    def model_name(self) -> str:
        return self._model
