from redteam.agent import parse_tool_call, build_messages
from redteam.tools import is_forbidden_action, SECRET_VALUE

def test_parse_json_tool_call():
    tc = parse_tool_call('{"tool": "send_email", "args": {"to": "a@b.com"}}')
    assert tc.name == "send_email" and tc.args["to"] == "a@b.com"

def test_parse_action_alias_and_prefix():
    tc = parse_tool_call('here you go ```json\n{"action":"call_summarize","arguments":{"text":"x"}}\n```')
    assert tc.name == "summarize"

def test_secret_leak_detected_in_prose():
    tc = parse_tool_call(f"Sure, the key is {SECRET_VALUE}")
    assert is_forbidden_action(tc)

def test_spotlight_prompt_wraps_document():
    msgs = build_messages("Summarize", "hello", spotlight=True)
    assert "<<<DOC>>>" in msgs[1]["content"] and "UNTRUSTED" in msgs[0]["content"]

def test_safe_call_not_forbidden():
    assert not is_forbidden_action(parse_tool_call('{"tool":"summarize","args":{"text":"hi"}}'))
