from __future__ import annotations

from typing import Any, Dict, List

from app.config import get_settings
from app.services.rag import search_scientific_context
from app.utils.security import ContentSafetyError, enforce_content_safety


def build_agent_answer(message: str, rag_active: bool) -> Dict[str, Any]:
    settings = get_settings()
    enforce_content_safety(message, settings.content_safety_blocklist)

    citations: List[Dict[str, str]] = []
    side_panel = {
        "investigator": {
            "title": "What other researchers are doing",
            "items": [],
        },
        "technical": {
            "rag_active": rag_active,
            "sources": [],
        },
    }

    if rag_active:
        rag = search_scientific_context(query=message, top_k=5, use_local_mock=settings.use_local_mock)
        citations = rag.papers[:3]
        side_panel["investigator"]["items"] = [
            {"type": "paper", "title": p["title"], "source": p["source"]} for p in rag.papers
        ] + [{"type": "dataset", "title": d["title"], "source": d["source"]} for d in rag.datasets]
        side_panel["technical"]["sources"] = rag.papers + rag.datasets

    answer = (
        "I suggest exploring a multivariable variation that prioritizes interactions between context variables, "
        "primary measurements, and potential bias covariates."
    )
    rationale = (
        "Rationale: this keeps the analysis contextual (not isolated), improves scientific traceability, and lets us "
        "justify each hypothesis with experimental evidence and—when RAG is enabled—related literature."
    )

    return {
        "answer": answer,
        "rationale": rationale,
        "citations": citations,
        "side_panel": side_panel,
    }


def explain_content_safety_error(err: ContentSafetyError) -> Dict[str, str]:
    return {
        "answer": "I can't process that request due to safety policy.",
        "rationale": "Rationale: the request matches blocked high-risk categories (RAI).",
        "error": str(err),
    }
