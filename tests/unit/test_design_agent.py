from src.agents.design_agent import _normalize_design


def test_normalize_design_fills_missing_keys():
    raw = {
        "overall_score": 42.0,
        "techniques_applied": [
            {"technique": "BVA"},
            {"coverage": "full"},  # missing technique
            "equivalence partitioning",  # non-dict entry
        ],
        "weak_tests": [{"test_case_id": "TC-1"}],
        "test_scores": [{"test_case_id": "TC-1", "score": 3}],
    }
    out = _normalize_design(raw)
    ta = out["techniques_applied"]
    assert ta[0] == {"technique": "BVA", "coverage": "n/a"}
    assert ta[1] == {"technique": "?", "coverage": "full"}
    assert ta[2] == {"technique": "equivalence partitioning", "coverage": "n/a"}
    wt = out["weak_tests"][0]
    assert wt["test_case_id"] == "TC-1" and wt["reason"] == ""
    assert out["test_scores"][0]["score"] == 3
    # missing top-level lists are present
    assert out["missing_techniques"] == []
    assert out["duplicate_tests"] == []
    assert out["recommendations"] == []


def test_normalize_design_rejects_non_dict():
    import pytest

    # Невалидный формат ответа LLM (не объект) должен бросать ошибку,
    # чтобы узел поймал её и quality_gate повторил агента (а не молча
    # генерировал пустой отчёт с score=0).
    with pytest.raises(ValueError):
        _normalize_design("not a dict")
    with pytest.raises(ValueError):
        _normalize_design(["a", "b"])
