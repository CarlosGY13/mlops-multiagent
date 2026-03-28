from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import AgentMessageRequest, AgentMessageResponse, DataFeedbackRequest, LiteratureSearchRequest, ResearchSearchRequest
from app.services.agent import build_agent_answer, build_data_feedback, build_search_insight, explain_content_safety_error
from app.services.rag import search_scientific_context
from app.utils.security import ContentSafetyError

router = APIRouter(prefix="/api/part3", tags=["part3-agent-ui"])


@router.post("/rag/search")
def rag_search(req: LiteratureSearchRequest):
    result = search_scientific_context(req.query, req.top_k, use_local_mock=False)
    return {
        "investigator": {
            "title": "What other researchers are doing",
            "papers": [{"title": p["title"], "source": p["source"]} for p in result.papers],
            "datasets": [{"title": d["title"], "source": d["source"]} for d in result.datasets],
        },
        "technical": {
            "papers": result.papers,
            "datasets": result.datasets,
            "indexing": "openalex_live_with_fallback",
        },
    }


@router.post("/agent/message", response_model=AgentMessageResponse)
def message(req: AgentMessageRequest) -> AgentMessageResponse:
    try:
        response = build_agent_answer(
            message=req.message,
            rag_active=req.rag_active,
            dataset_id=req.dataset_id,
            session_id=req.session_id,
            user_context=req.user_context,
        )
    except ContentSafetyError as err:
        safe = explain_content_safety_error(err)
        raise HTTPException(status_code=400, detail=safe)

    return AgentMessageResponse(**response)


@router.post("/data/feedback")
def data_feedback(req: DataFeedbackRequest):
    try:
        return build_data_feedback(dataset_id=req.dataset_id, session_id=req.session_id, user_context=req.user_context)
    except ContentSafetyError as err:
        safe = explain_content_safety_error(err)
        raise HTTPException(status_code=400, detail=safe)


@router.post("/search")
def research_search(req: ResearchSearchRequest):
    try:
        rag = search_scientific_context(req.query, req.top_k, use_local_mock=False)
        insight = build_search_insight(
            query=req.query,
            papers=rag.papers,
            datasets=rag.datasets,
            dataset_id=req.dataset_id,
            session_id=req.session_id,
            user_context=req.user_context,
        )
        return {
            "investigator": insight,
            "technical": {"papers": rag.papers, "datasets": rag.datasets},
        }
    except ContentSafetyError as err:
        safe = explain_content_safety_error(err)
        raise HTTPException(status_code=400, detail=safe)
