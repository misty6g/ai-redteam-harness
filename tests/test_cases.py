from redteam.cases import build_suite, ATTACKER_EMAIL

def test_suite_sizes_and_determinism():
    a = build_suite(seed=1337)
    b = build_suite(seed=1337)
    assert [c.id for c in a] == [c.id for c in b]  # reproducible order
    inj = [c for c in a if c.kind == "injection"]
    ben = [c for c in a if c.kind == "benign"]
    assert len(inj) == 100 and len(ben) == 32
    assert len(a) == 132

def test_injection_cases_carry_payload():
    a = build_suite(seed=1337)
    overt = [c for c in a if c.kind == "injection" and c.overt]
    subtle = [c for c in a if c.kind == "injection" and not c.overt]
    assert len(overt) == 50 and len(subtle) == 50
    for c in a:
        if c.kind == "injection":
            assert c.objective is not None
        else:
            assert c.expected_safe_tool is not None

def test_benign_docs_have_no_attacker_markers():
    for c in build_suite(seed=1337):
        if c.kind == "benign":
            assert ATTACKER_EMAIL not in c.document
            assert "evil-collector" not in c.document
