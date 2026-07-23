"""Unit tests for request correlation forensics helpers."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from request_correlation import (
    OrphanFrameMeter,
    cmd_kind,
    corr_log_enabled,
    format_corr,
    summarize_first_tokens,
)


class CorrConfigTests(unittest.TestCase):
    def test_corr_log_defaults_on(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DFLASH_CORR_LOG", None)
            self.assertTrue(corr_log_enabled())

    def test_corr_log_off(self) -> None:
        with patch.dict(os.environ, {"DFLASH_CORR_LOG": "0"}):
            self.assertFalse(corr_log_enabled())


class FormatCorrTests(unittest.TestCase):
    def test_joins_scope_slot_req(self) -> None:
        line = format_corr(
            "gen_start",
            scope="8db4b93b-65da-4582-bcf7-b84fc0e0f9db:69c563073027dcab",
            slot=0,
            req_id=42,
            cmd="RESTORE_CHAIN",
            gen_len=128,
        )
        self.assertIn("[corr] gen_start", line)
        self.assertIn("slot=0", line)
        self.assertIn("req=42", line)
        self.assertIn("cmd=RESTORE_CHAIN", line)
        self.assertIn("8db4b93b", line)

    def test_short_scope_keeps_fingerprint(self) -> None:
        line = format_corr(
            "x",
            scope="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:deadbeefcafebabe",
        )
        self.assertIn(":deadbeefcafebabe", line)


class CmdKindTests(unittest.TestCase):
    def test_peels_req_slot(self) -> None:
        self.assertEqual(
            cmd_kind("REQ 9 SLOT 1 RESTORE_CHAIN -1 4 /tmp/p.bin 12 128\n"),
            "RESTORE_CHAIN",
        )

    def test_generate(self) -> None:
        self.assertEqual(cmd_kind("START /tmp/p.bin 64"), "GENERATE")


class FirstTokensTests(unittest.TestCase):
    def test_ids_without_tokenizer(self) -> None:
        s = summarize_first_tokens([10, 20, 30, 40], n=3)
        self.assertEqual(s["n_tok"], 4)
        self.assertEqual(s["first_ids"], "10,20,30")
        self.assertNotIn("first_text", s)

    def test_detok_prefix(self) -> None:
        class _Tok:
            def decode(self, ids, skip_special_tokens=False):
                return "I'll research Tesla Level 4"

        s = summarize_first_tokens([1, 2, 3], tokenizer=_Tok(), n=3)
        self.assertIn("Tesla", s["first_text"])


class OrphanMeterTests(unittest.TestCase):
    def test_rate_limits(self) -> None:
        meter = OrphanFrameMeter(debounce_sec=60.0)
        with patch.dict(os.environ, {"DFLASH_CORR_LOG": "1"}), patch(
            "request_correlation.print"
        ) as mock_print:
            meter.note(7, "tag")
            meter.note(7, "tag")
            meter.note(8, "cont")
            self.assertEqual(mock_print.call_count, 1)
            self.assertIn("demux_orphan", mock_print.call_args[0][0])

    def test_after_collect_window_logs_distinct_event(self) -> None:
        meter = OrphanFrameMeter(debounce_sec=0.0)
        with patch.dict(os.environ, {"DFLASH_CORR_LOG": "1"}), patch(
            "request_correlation.print"
        ) as mock_print:
            meter.mark_after_collect(window_sec=5.0)
            meter.note(9, "tag")
            self.assertEqual(meter.after_collect_orphans, 1)
            self.assertIn("demux_orphan_after_collect", mock_print.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
