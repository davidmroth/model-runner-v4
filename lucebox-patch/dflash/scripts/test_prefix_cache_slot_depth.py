"""Unit tests: prefix cache slot depth vs lookup cut (cross-session safety)."""
import unittest
from unittest.mock import MagicMock, patch

from prefix_cache import PrefixCache, hash_prefix, resolve_cache_scope, scope_skips_prefix_snap


class _FakeTokenizer:
    """Minimal Qwen-style markers so PrefixCache enables itself."""

    _MARKERS = {
        "<|im_end|>": [1],
        "<|im_start|>": [2],
        "system": [3],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._MARKERS.get(text, [ord(c) for c in text]))


class PrefixCacheSlotDepthTests(unittest.TestCase):
    def _make_cache(self) -> PrefixCache:
        pc = PrefixCache(
            daemon_stdin=MagicMock(),
            await_reply=MagicMock(),
            daemon_lock=MagicMock(),
            tokenizer=_FakeTokenizer(),
            kv_k_type="f16",
            fa_window=0,
            cap=4,
        )
        self.assertFalse(pc.disabled)
        return pc

    def _scoped_entry(self, pc: PrefixCache, ids: list[int], cut: int, scope: str):
        key = hash_prefix(ids[:cut], pc.kv_k_type, pc.fa_window, scope)
        return (scope, key)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_lookup_rejects_stale_shallow_hash(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]
        scope = "sess-a"

        entry_key = self._scoped_entry(pc, ids, 376, scope)
        pc.entries[entry_key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 2411
        pc._slot_scope[0] = scope

        self.assertIsNone(pc.lookup(ids, scope=scope))
        self.assertNotIn(entry_key, pc.entries)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_lookup_accepts_matching_depth(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]
        scope = "sess-a"

        entry_key = self._scoped_entry(pc, ids, 376, scope)
        pc.entries[entry_key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376
        pc._slot_scope[0] = scope

        self.assertEqual(pc.lookup(ids, scope=scope), (0, 376))

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_lookup_rejects_cross_scope_slot(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]

        bench_key = self._scoped_entry(pc, ids, 376, "benchmark")
        pc.entries[bench_key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376
        pc._slot_scope[0] = "benchmark"

        self.assertIsNone(pc.lookup(ids, scope="agent-session"))
        self.assertIn(bench_key, pc.entries)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_confirm_evicts_stale_keys_for_slot(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(500))
        mock_bounds.return_value = [100, 376, 480]
        scope = "sess-a"

        shallow = self._scoped_entry(pc, ids, 376, scope)
        deep = self._scoped_entry(pc, ids, 480, scope)
        pc.entries[shallow] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376
        pc._slot_scope[0] = scope

        pc.confirm_inline_snap(0, 480, ids, scope=scope)

        self.assertNotIn(shallow, pc.entries)
        self.assertIn(deep, pc.entries)
        self.assertEqual(pc.entries[deep], 0)
        self.assertEqual(pc._slot_prefix_len[0], 480)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_abort_inline_snap_purges_reuse_slot_mappings(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]
        scope = "sess-a"

        entry_key = self._scoped_entry(pc, ids, 376, scope)
        pc.entries[entry_key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376
        pc._slot_scope[0] = scope

        prep = pc.prepare_inline_snap(ids, reuse_slot=0, scope=scope)
        self.assertIsNotNone(prep)
        slot, _ = prep
        pc.abort_inline_snap(slot, scope=scope)

        self.assertNotIn(entry_key, pc.entries)
        self.assertNotIn(0, pc._slot_prefix_len)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_ephemeral_prepare_skips_inline_snap(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]
        ephemeral = resolve_cache_scope(conversation_id=None, prompt_ids=ids)

        self.assertTrue(scope_skips_prefix_snap(ephemeral))
        self.assertIsNone(pc.prepare_inline_snap(ids, scope=ephemeral))
        self.assertEqual(len(pc.entries), 0)

    @patch("prefix_cache.find_all_boundaries_markers")
    def test_ephemeral_confirm_does_not_evict_conversation_slot(self, mock_bounds):
        pc = self._make_cache()
        ids = list(range(400))
        mock_bounds.return_value = [50, 376]
        conv_scope = "e078410b-cd79-4685-8b7b-8d760dc370e8"
        conv_key = self._scoped_entry(pc, ids, 376, conv_scope)

        pc.entries[conv_key] = 0
        pc._populated_slots.add(0)
        pc._slot_prefix_len[0] = 376
        pc._slot_scope[0] = conv_scope

        ephemeral = resolve_cache_scope(conversation_id=None, prompt_ids=ids[:60])
        pc.confirm_inline_snap(0, 59, ids[:60], scope=ephemeral)

        self.assertIn(conv_key, pc.entries)
        self.assertEqual(pc._slot_prefix_len[0], 376)
        self.assertEqual(pc._slot_scope[0], conv_scope)

    def test_abort_full_snap_purges_stale_lookup(self):
        pc = self._make_cache()
        pc.init_full_cache(1)
        scope = "sess-a"
        old_ids = list(range(200))
        old_key = hash_prefix(old_ids, pc.kv_k_type, pc.fa_window, scope)
        entry_key = (scope, old_key)
        pc.full_entries[entry_key] = (pc._full_slot_base, "/tmp/old.bin", 100)

        new_ids = list(range(300))
        prep = pc.prepare_full_snap(new_ids, scope=scope)
        self.assertIsNotNone(prep)
        slot, _ = prep
        self.assertEqual(pc._full_pending_evict_key, entry_key)
        pc.abort_full_snap(slot)

        self.assertIsNone(pc.lookup_full(old_ids, scope=scope))


class CacheScopeTests(unittest.TestCase):
    def test_conversation_id_scopes_sessions(self):
        ids = list(range(100))
        a = resolve_cache_scope(conversation_id="conv-1", prompt_ids=ids)
        b = resolve_cache_scope(conversation_id="conv-2", prompt_ids=ids)
        self.assertNotEqual(a, b)

    def test_ephemeral_scope_isolates_unlabeled_requests(self):
        ids_a = list(range(50))
        ids_b = list(range(51))
        a = resolve_cache_scope(conversation_id=None, prompt_ids=ids_a)
        b = resolve_cache_scope(conversation_id=None, prompt_ids=ids_b)
        self.assertTrue(a.startswith("ephemeral:"))
        self.assertTrue(b.startswith("ephemeral:"))
        self.assertNotEqual(a, b)

    def test_same_ephemeral_prompt_reuses_scope(self):
        ids = list(range(80))
        a = resolve_cache_scope(conversation_id=None, prompt_ids=ids, tools_fingerprint="fp1")
        b = resolve_cache_scope(conversation_id=None, prompt_ids=ids, tools_fingerprint="fp1")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
