"""DeepEval metrics configuration.

WHY six metrics: Faithfulness + Answer Relevancy validate the answer quality;
the two Contextual metrics validate the retrieval layer; Hallucination catches
fabricated facts not grounded in context; Toxicity ensures safe output.
All six together give end-to-end coverage of the full RAG pipeline.
"""
from __future__ import annotations

import os
from typing import List


def _gpt_model(model: str):
    """Build a GPTModel with the API key explicitly injected.

    WHY explicit key: DeepEval 4.x resolves the OpenAI key at GPTModel
    instantiation time from os.environ. Passing it directly here avoids
    any env-lookup timing issue and the resulting AuthenticationError.
    """
    from deepeval.models.llms.openai_model import GPTModel
    return GPTModel(model=model, api_key=os.environ.get("OPENAI_API_KEY", ""))


def build_metrics(model: str = "gpt-4o-mini") -> List:
    """Construct DeepEval metric instances at the thresholds from §13.2.

    WHY a builder function: deepeval imports torch transitively and is
    expensive to import. Lazy imports keep `python -m api.main` fast.
    WHY include_reason=True: surfaces the LLM's explanation for each
    metric score in the evaluation report, making failures debuggable.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        ToxicityMetric,
    )
    m = _gpt_model(model)
    return [
        FaithfulnessMetric(threshold=0.70, model=m, include_reason=True),
        AnswerRelevancyMetric(threshold=0.70, model=m, include_reason=True),
        ContextualRecallMetric(threshold=0.60, model=m, include_reason=True),
        ContextualPrecisionMetric(threshold=0.50, model=m, include_reason=True),
        HallucinationMetric(threshold=0.50, model=m, include_reason=True),
        ToxicityMetric(threshold=0.10, model=m, include_reason=True),
    ]


def build_rag_metrics(model: str = "gpt-4o-mini") -> List:
    """RAG-only subset (Faithfulness + both Contextual) for retrieval-focused tests.

    WHY separate: repair_cost and total_loss tests hit CSV tools where
    retrieval_context is structured data, not policy prose. Running the
    full 6-metric suite on those cases inflates HallucinationMetric
    false-positives because numeric context is terse by design.
    """
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    m = _gpt_model(model)
    return [
        FaithfulnessMetric(threshold=0.70, model=m, include_reason=True),
        ContextualRecallMetric(threshold=0.60, model=m, include_reason=True),
        ContextualPrecisionMetric(threshold=0.50, model=m, include_reason=True),
    ]
