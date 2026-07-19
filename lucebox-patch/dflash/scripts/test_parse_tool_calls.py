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

# Decode stopped after </function> but before </tool_call>: the opener has no
# matching close, so the complete-block regex misses it and (pre-fix) the literal
# <tool_call> tag leaked into assistant content.
UNCLOSED_TOOL_CALL = """Sure, let me put that together.
<tool_call>
<function=create_briefing>
<parameter=title>
Daily AI News Digest
</parameter>
</function>"""


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

    def test_unclosed_tool_call_opener_not_leaked(self):
        """Recovered call from an unclosed block must not leak the <tool_call> tag."""
        cleaned, tcs = parse_tool_calls(UNCLOSED_TOOL_CALL)
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "create_briefing")
        args = json.loads(tcs[0]["function"]["arguments"])
        self.assertEqual(args.get("title"), "Daily AI News Digest")
        self.assertNotIn("<tool_call>", cleaned)
        self.assertNotIn("</tool_call>", cleaned)
        # Leading prose is preserved.
        self.assertIn("let me put that together", cleaned)

    def test_prose_without_tools_unchanged(self):
        text = "Use <function> declarations in JavaScript."
        cleaned, tcs = parse_tool_calls(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(tcs, [])

    def test_truncated_function_name_not_parsed_as_tool(self):
        """Missing `>` after name must not swallow `<parameter=…>` into the name."""
        text = (
            "<tool_call>\n"
            "<function=browser_navigate\n"
            "<parameter=url>\n"
            "https://example.com\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        cleaned, tcs = parse_tool_calls(text)
        self.assertEqual(tcs, [])
        # Incomplete tags remain as content rather than a poison tool name.
        self.assertIn("browser_navigate", cleaned)


if __name__ == "__main__":
    unittest.main()
