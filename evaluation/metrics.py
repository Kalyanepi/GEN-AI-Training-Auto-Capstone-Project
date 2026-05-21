"""DeepEval metrics configuration.

WHY four metrics: Faithfulness + Answer Relevancy validate the answer; the
two Contextual metrics validate the retrieval layer underneath. All four
together give end-to-end coverage of the RAG pipeline.
"""
from __future__ import annotations

from typing import List


def build_metrics(model: str = "gpt-4o-mini") -> List:
    """Construct DeepEval metric instances at the thresholds from §13.2.

    WHY a builder function: deepeval imports torch transitively and is
    expensive to import. Lazy imports keep `python -m api.main` fast.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    return [
        FaithfulnessMetric(threshold=0.85, model=model),
        AnswerRelevancyMetric(threshold=0.80, model=model),
        ContextualRecallMetric(threshold=0.75, model=model),
        ContextualPrecisionMetric(threshold=0.70, model=model),
    ]
