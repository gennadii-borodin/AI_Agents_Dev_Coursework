import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.llm_provider import RouterAIProvider
from src.config import get_settings
from src.tools.sql_tool import get_all_test_cases

settings = get_settings()
llm = RouterAIProvider(settings)

tcs = get_all_test_cases()[:10]

import json
tc_json = json.dumps([
    {"id": tc["test_case_id"], "req": tc["req"], "title": tc["title"][:60], "type": tc["test_type"], "quality": tc["design_quality"], "review": tc["qa_review"]}
    for tc in tcs
], ensure_ascii=True)

msg = f"Оцени дизайн тестов:\n{tc_json}\nВерни JSON."

print(f"Prompt length: {len(msg)} chars, ~{len(msg)//4} tokens")

response = llm.chat_completion(
    messages=[
        {"role": "system", "content": "Верни JSON с overall_score, weak_tests, recommendations."},
        {"role": "user", "content": msg},
    ],
    model=settings.model_senior,
    temperature=0.1,
    json_mode=True,
)
print(f"Response length: {len(response) if response else 0}")
print(f"Response: {response[:300] if response else 'EMPTY'}")
