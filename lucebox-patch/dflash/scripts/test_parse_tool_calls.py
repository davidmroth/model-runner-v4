"""Unit tests for parse_tool_calls — including bare <function= blocks."""
import json
import unittest

# Import only parser helpers (avoid FastAPI / daemon deps).
from server_tools import parse_tool_calls

FLOCK_BARE = """I don't have solid, verified information. Let me search for it.

<function=web>
<parameter=query>
Flock camera what is it product

<parameter=tool>
search
"""

WRAPPED = """<tool_call>
<function=terminal>
<parameter=command>
echo hello
</function>
</tool_call>"""


class ParseToolCallsTests(unittest.TestCase):
    def test_bare_function_block_parsed(self):
        cleaned, tcs = parse_tool_calls(FLOCK_BARE)
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "web")
        args = json.loads(tcs[0]["function"]["arguments"])
        self.assertIn("query", args)
        self.assertNotIn("<function=", cleaned)

    def test_wrapped_tool_call_still_works(self):
        cleaned, tcs = parse_tool_calls(WRAPPED)
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "terminal")
        self.assertNotIn("<tool_call>", cleaned)

    def test_prose_without_tools_unchanged(self):
        text = "Use <function> declarations in JavaScript."
        cleaned, tcs = parse_tool_calls(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(tcs, [])


if __name__ == "__main__":
    unittest.main()
