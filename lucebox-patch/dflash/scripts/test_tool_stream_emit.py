"""Unit tests for OpenAI-style incremental tool_calls emit."""
import json
import unittest

from tool_stream_emit import ToolStreamState, feed_tool_stream, should_skip_final_tool_emit


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
]


def _concat_args(chunks: list) -> str:
    out = ""
    for c in chunks:
        fn = c["tool_calls"][0].get("function") or {}
        if "arguments" in fn:
            out += fn["arguments"]
    return out


class ToolStreamEmitTests(unittest.TestCase):
    def test_name_then_args_suffixes(self):
        state = ToolStreamState()
        chunks = []
        chunks += feed_tool_stream("<tool_call>\n<function=web_search>", state, TOOLS)
        self.assertEqual(len(chunks), 1)
        tc = chunks[0]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "web_search")
        self.assertEqual(tc["function"]["arguments"], "")
        self.assertTrue(tc["id"].startswith("call_"))

        buf = (
            "<tool_call>\n<function=web_search>\n"
            "<parameter=query>AI news</parameter>\n"
        )
        chunks += feed_tool_stream(buf, state, TOOLS)
        self.assertIn(
            '{"query": "AI news"',
            _concat_args(chunks),
        )

        buf += "<parameter=limit>5</parameter>\n</function>\n</tool_call>"
        chunks += feed_tool_stream(buf, state, TOOLS)
        args = _concat_args(chunks)
        self.assertEqual(json.loads(args), {"query": "AI news", "limit": 5})
        self.assertTrue(should_skip_final_tool_emit(state))
        self.assertEqual(state.index, 1)
        self.assertIsNone(state.name)

    def test_no_emit_until_name_closed(self):
        state = ToolStreamState()
        chunks = feed_tool_stream("<tool_call>\n<function=web_sear", state, TOOLS)
        self.assertEqual(chunks, [])
        self.assertFalse(state.streamed_any)

    def test_idempotent_on_same_buffer(self):
        state = ToolStreamState()
        buf = (
            "<tool_call><function=web_search>"
            "<parameter=query>x</parameter></function></tool_call>"
        )
        a = feed_tool_stream(buf, state, TOOLS)
        b = feed_tool_stream(buf, state, TOOLS)
        self.assertGreaterEqual(len(a), 1)
        self.assertEqual(b, [])
        self.assertEqual(json.loads(_concat_args(a)), {"query": "x"})

    def test_multi_tool_indexes(self):
        state = ToolStreamState()
        buf = (
            "<tool_call><function=web_search>"
            "<parameter=query>a</parameter></function></tool_call>"
            "<tool_call><function=web_search>"
            "<parameter=query>b</parameter></function></tool_call>"
        )
        chunks = feed_tool_stream(buf, state, TOOLS)
        names = [
            c["tool_calls"][0]
            for c in chunks
            if c["tool_calls"][0].get("function", {}).get("name")
        ]
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0]["index"], 0)
        self.assertEqual(names[1]["index"], 1)
        self.assertEqual(state.index, 2)


if __name__ == "__main__":
    unittest.main()
