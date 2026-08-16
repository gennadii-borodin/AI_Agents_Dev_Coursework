from typing import Any, Optional

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    requirement_id: str
    title: str
    requirement_text: str
    category: str
    priority: str
    qa_requirements_review: str = ""
    rejection_reason: str = ""


class TestCase(BaseModel):
    test_case_id: str
    req: str
    title: str
    description: str
    preconditions: str
    test_data: str
    steps: str
    expected_result: str
    priority: str
    test_type: str
    design_quality: str
    qa_review: str
    review_comment: str = ""


class Violation(BaseModel):
    rule_id: str
    severity: str
    test_case_id: str
    description: str
    auto_fixable: bool = False


class CoverageReport(BaseModel):
    total_coverage: float
    critical_coverage: float
    matrix: list[dict[str, Any]] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(default_factory=list)
    tests_without_requirements: list[str] = Field(default_factory=list)
    indirect_coverage: list[dict[str, str]] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    residual_risk: str = ""


class DesignReport(BaseModel):
    overall_score: float
    techniques_applied: list[dict[str, str]] = Field(default_factory=list)
    missing_techniques: list[str] = Field(default_factory=list)
    weak_tests: list[dict[str, str]] = Field(default_factory=list)
    duplicate_tests: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    test_scores: list[dict[str, Any]] = Field(default_factory=list)


class StandardsReport(BaseModel):
    compliance_percentage: float
    violations: list[dict[str, Any]] = Field(default_factory=list)
    blocking_violations: list[str] = Field(default_factory=list)
    auto_fix_available: list[str] = Field(default_factory=list)
    human_review_required: list[str] = Field(default_factory=list)
    # Флаг частичного анализа: часть чанков не проанализировалась
    # (исчерпаны retries LLM). compliance в этом случае считается только по
    # проанализированным ТК и не завышается.
    partial: bool = False
    failed_chunks: int = 0
    analyzed_test_cases: int = 0


class ReviewState(BaseModel):
    user_query: str
    scenario: str = ""
    requirement_ids: Optional[list[str]] = None
    agents_to_run: list[str] = Field(default_factory=list)

    requirements: list[Requirement] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(default_factory=list)
    standards_text: str = ""

    coverage_report: Optional[CoverageReport] = None
    design_report: Optional[DesignReport] = None
    standards_report: Optional[StandardsReport] = None

    rag_results: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    sql_results: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    errors: list[str] = Field(default_factory=list)
    current_step: str = ""

    # Поля оркестрации/наблюдаемости (revью §4, Этап 4).
    task_id: str = ""
    user_goal: str = ""
    input_artifact_refs: list[str] = Field(default_factory=list)
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    violations_ref: str = ""
    unresolved_questions: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 0
    final_answer: str = ""
