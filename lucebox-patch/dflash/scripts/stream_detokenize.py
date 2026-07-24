"""Incremental detokenization that preserves multi-token UTF-8 emoji.

Qwen (and other byte-fallback BPEs) split many emoji into 2–3 token ids.
``tokenizer.decode([id])`` for each fragment returns U+FFFD; only the
cumulative ``decode(ids[:n])`` yields the real glyph once the sequence is
complete.

If those U+FFFD pieces are streamed to the client, they are permanent — the
API cannot retract them when the emoji later completes.  This helper holds
trailing U+FFFD until a later push (or ``finish()``) resolves them.
"""
from __future__ import annotations

from typing import Any


_REPL = "\ufffd"


def stable_detokenized_prefix(text: str) -> str:
    """Drop trailing U+FFFD code points (incomplete byte-token sequences)."""
    end = len(text)
    while end > 0 and text[end - 1] == _REPL:
        end -= 1
    return text[:end]


class IncrementalDetokenizer:
    """Push token ids; receive only newly stable Unicode text."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        skip_special_tokens: bool = True,
        max_token_id: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._skip_special = skip_special_tokens
        self._ids: list[int] = []
        self._emitted = ""
        # When set, ids outside ``[0, max_token_id]`` are skipped (demux
        # cross-talk / bare-pipe tag frames must not crash ASGI streams).
        if max_token_id is not None:
            self._max_token_id = int(max_token_id)
        else:
            self._max_token_id = self._infer_max_token_id(tokenizer)

    @staticmethod
    def _infer_max_token_id(tokenizer: Any) -> int | None:
        for attr in ("vocab_size", "get_vocab_size"):
            try:
                val = getattr(tokenizer, attr)
                n = int(val() if callable(val) else val)
            except Exception:
                continue
            if n > 0:
                return n - 1
        return None

    def _is_vocab_id(self, tok_id: int) -> bool:
        if tok_id < 0:
            return False
        if self._max_token_id is not None and tok_id > self._max_token_id:
            return False
        return True

    def push(self, tok_id: int) -> str:
        tid = int(tok_id)
        if not self._is_vocab_id(tid):
            return ""
        self._ids.append(tid)
        try:
            full = self._tokenizer.decode(
                self._ids, skip_special_tokens=self._skip_special
            )
        except (OverflowError, ValueError, TypeError):
            # Drop the bad id that rust/HF rejected and keep streaming.
            self._ids.pop()
            return ""
        stable = stable_detokenized_prefix(full)
        if not stable.startswith(self._emitted):
            # Prefix invalidated (should be rare if we always withhold U+FFFD).
            # Emit the full stable string as a best-effort correction delta
            # only for the suffix beyond the common prefix.
            common = 0
            limit = min(len(stable), len(self._emitted))
            while common < limit and stable[common] == self._emitted[common]:
                common += 1
            delta = stable[common:]
            self._emitted = stable
            return delta
        delta = stable[len(self._emitted) :]
        self._emitted = stable
        return delta

    def finish(self) -> str:
        """Flush any remaining text, including unresolved U+FFFD at EOS."""
        if not self._ids:
            return ""
        try:
            full = self._tokenizer.decode(
                self._ids, skip_special_tokens=self._skip_special
            )
        except (OverflowError, ValueError, TypeError):
            return ""
        if full.startswith(self._emitted):
            delta = full[len(self._emitted) :]
        else:
            common = 0
            limit = min(len(full), len(self._emitted))
            while common < limit and full[common] == self._emitted[common]:
                common += 1
            delta = full[common:]
        self._emitted = full
        return delta
