"""Unit tests for IncrementalDetokenizer (no HF download)."""
from __future__ import annotations

import unittest

from stream_detokenize import IncrementalDetokenizer, stable_detokenized_prefix


class FakeByteFallbackTokenizer:
    """Mimics Qwen: emoji split across byte tokens that decode to U+FFFD alone."""

    def __init__(self, table: dict[tuple[int, ...], str]) -> None:
        self._table = table

    def decode(self, ids, skip_special_tokens=True):
        key = tuple(int(i) for i in ids)
        if key in self._table:
            return self._table[key]
        # Incomplete prefix → replacement chars (one per id), like HF byte tokens.
        return "\ufffd" * len(key)


class IncrementalDetokenizerTests(unittest.TestCase):
    def test_stable_prefix_strips_trailing_fffd(self):
        self.assertEqual(stable_detokenized_prefix("ab\ufffd\ufffd"), "ab")
        self.assertEqual(stable_detokenized_prefix("\ufffd"), "")
        self.assertEqual(stable_detokenized_prefix("ok"), "ok")

    def test_emoji_held_until_complete(self):
        # 👍 = ids 9008,239,235 in real Qwen; fake the same pattern.
        tok = FakeByteFallbackTokenizer({
            (1, 2, 3): "👍",
            (1, 2, 3, 4): "👍!",
        })
        dec = IncrementalDetokenizer(tok)
        self.assertEqual(dec.push(1), "")
        self.assertEqual(dec.push(2), "")
        self.assertEqual(dec.push(3), "👍")
        self.assertEqual(dec.push(4), "!")
        self.assertEqual(dec.finish(), "")

    def test_single_token_emoji_emits_immediately(self):
        tok = FakeByteFallbackTokenizer({(9,): "✅"})
        dec = IncrementalDetokenizer(tok)
        self.assertEqual(dec.push(9), "✅")
        self.assertEqual(dec.finish(), "")

    def test_ascii_streams_token_by_token(self):
        tok = FakeByteFallbackTokenizer({
            (10,): "H",
            (10, 11): "Hi",
            (10, 11, 12): "Hi!",
        })
        dec = IncrementalDetokenizer(tok)
        self.assertEqual(dec.push(10), "H")
        self.assertEqual(dec.push(11), "i")
        self.assertEqual(dec.push(12), "!")

    def test_finish_flushes_unresolved_fffd(self):
        tok = FakeByteFallbackTokenizer({})  # always FFFD until unknown
        dec = IncrementalDetokenizer(tok)
        self.assertEqual(dec.push(1), "")
        # At EOS, flush whatever decode returns (true garbage / truncated).
        self.assertEqual(dec.finish(), "\ufffd")

    def test_negative_and_oob_ids_skipped(self):
        tok = FakeByteFallbackTokenizer({(10,): "H", (10, 11): "Hi"})
        dec = IncrementalDetokenizer(tok, max_token_id=100)
        self.assertEqual(dec.push(-2), "")  # tag marker
        self.assertEqual(dec.push(10**12), "")  # overflow-class id
        self.assertEqual(dec.push(10), "H")
        self.assertEqual(dec.push(11), "i")

    def test_overflow_from_tokenizer_does_not_raise(self):
        class BoomTok:
            def decode(self, ids, skip_special_tokens=True):
                if any(i > 40 for i in ids):
                    raise OverflowError("out of range integral type conversion attempted")
                return "x" * len(ids)

        # No vocab_size → cannot pre-filter; decode OverflowError must be caught.
        dec = IncrementalDetokenizer(BoomTok(), max_token_id=None)
        self.assertEqual(dec.push(1), "x")
        self.assertEqual(dec.push(99), "")  # caught, prior state kept
        self.assertEqual(dec.push(2), "x")


if __name__ == "__main__":
    unittest.main()
