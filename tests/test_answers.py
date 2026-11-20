"""Answer accuracy on the eval set (RAG-8): the generated answer contains the expected keywords.

Calls the Foundry LLM, so these are marked `llm` and skipped when the settings are not set.
"""

import pytest

from rag import evaluation as ev
from rag import pipeline
from rag.config import TOP_K

# 11 of 13 cases. Measured: vector 92%, hybrid 92%, rerank 100% (eval/results.md)
ACCURACY_THRESHOLD = 0.8

pytestmark = pytest.mark.llm


@pytest.mark.parametrize("mode", pipeline.MODES)
def test_answer_accuracy_on_the_eval_set(mode, real_collection, request):
    if mode == "rerank":  # only this mode needs the cross-encoder; the others must not skip for it
        request.getfixturevalue("cross_encoder")
    collection, embed = real_collection

    def ask(question):
        return pipeline.answer(question, k=TOP_K, mode=mode, collection=collection, embedder=embed)[
            "answer"
        ]

    results = ev.evaluate_answers(ev.load_cases(), ask)
    wrong = {r.case.id: r.missing for r in results if not r.passed}
    print(f"\n{mode}: accuracy {ev.accuracy(results):.0%}, missing keywords: {wrong}")
    assert ev.accuracy(results) >= ACCURACY_THRESHOLD, wrong


def test_the_unanswerable_question_gets_the_fallback_not_an_invented_answer(real_collection):
    collection, embed = real_collection
    case = next(c for c in ev.load_cases() if c.category == "unanswerable")
    out = pipeline.answer(case.question, collection=collection, embedder=embed)
    assert ev.missing_keywords(out["answer"], case) == []
    assert out["sources"] == []
