from src.agents.coverage_agent import _recompute_coverage
from src.agents.standards_agent import load_standards_rules, rule_classification
from src.prompts import build_agent_system_prompt, build_json_schema


def test_build_json_schema_structure():
    schema = build_json_schema("coverage_agent")
    assert schema["type"] == "object"
    assert "total_coverage" in schema["properties"]
    assert schema["properties"]["total_coverage"]["type"] == "number"
    assert "matrix" in schema["required"]


def test_build_agent_system_prompt_includes_few_shot():
    prompt = build_agent_system_prompt("coverage_agent")
    assert "Покрытие требований" in prompt
    assert "Примеры" in prompt
    assert "REQ-001" in prompt


def test_load_standards_rules_blocking():
    rules = load_standards_rules()["rules"]
    ids = {r["id"] for r in rules}
    assert "QA-TEST-010" in ids
    blocking, auto_fix = rule_classification()
    assert "QA-TEST-010" in blocking
    assert "QA-TEST-005" in auto_fix
    assert "QA-TEST-010" not in auto_fix


def test_recompute_coverage_from_matrix():
    data = {
        "matrix": [
            {"requirement_id": "REQ-1", "priority": "Critical", "weight": 3.0, "covered": True},
            {"requirement_id": "REQ-2", "priority": "High", "weight": 2.0, "covered": False},
        ],
        "total_coverage": 999.0,
        "critical_coverage": 999.0,
        "residual_risk": "low",
    }
    out = _recompute_coverage(data)
    # covered weight = 3 of total 5 -> 60.0
    assert out["total_coverage"] == 60.0
    # critical covered 3 of 3 -> 100.0
    assert out["critical_coverage"] == 100.0
    assert out["residual_risk"] == "high"  # critical_coverage < 100? no, 100 -> but total<80


def test_recompute_coverage_keeps_llm_values_when_no_matrix():
    data = {"total_coverage": 77.0, "critical_coverage": 100.0, "residual_risk": "low"}
    assert _recompute_coverage(data)["total_coverage"] == 77.0


def test_recompute_coverage_never_zero_on_degenerate_priority():
    # LLM вернул приоритет в другом регистре/пустой или weight=0:
    # покрытие не должно превращаться в 0% при наличии отмеченных требований.
    data = {
        "matrix": [
            {"requirement_id": "REQ-1", "priority": "critical", "covered": True},
            {"requirement_id": "REQ-2", "priority": "", "weight": 0, "covered": True},
        ]
    }
    out = _recompute_coverage(data)
    assert 0.0 < out["total_coverage"] <= 100.0
