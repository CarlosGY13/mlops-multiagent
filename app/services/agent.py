from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.dataset_context import dataset_context_json
from app.services.foundry_openai import FoundryOpenAIConfig, chat_completions, is_configured
from app.services.rag import search_scientific_context
from app.services.session_state import append_history, get_effective_context, get_history, set_user_context
from app.utils.security import ContentSafetyError, enforce_content_safety


def _foundry_cfg() -> FoundryOpenAIConfig:
    s = get_settings()
    return FoundryOpenAIConfig(
        base_url=s.foundry_openai_base_url,
        api_key=s.foundry_openai_api_key,
        model=s.foundry_openai_model,
        azure_endpoint=s.foundry_azure_openai_endpoint,
        azure_deployment=s.foundry_azure_openai_deployment,
        azure_api_version=s.foundry_azure_openai_api_version,
    )


def _parse_answer(text: str) -> Tuple[str, str]:
    t = (text or "").strip()
    if not t:
        return "", ""

    # Expected format:
    # ANSWER:\n...\n\nRATIONALE:\n...
    upper = t.upper()
    if "ANSWER:" in upper and "RATIONALE:" in upper:
        # Find case-insensitive markers
        a_idx = upper.find("ANSWER:")
        r_idx = upper.find("RATIONALE:")
        ans = t[a_idx + len("ANSWER:") : r_idx].strip()
        rat = t[r_idx + len("RATIONALE:") :].strip()
        return ans, rat

    # Fallback: treat full text as answer.
    return t, "Rationale: (not provided)"


def _base_system_prompt() -> str:
    return (
        "You are Lab Notebook AI: a scientific reasoning assistant for non-technical researchers. "
        "You help interpret experiments and datasets, and suggest multivariable variations with explicit rationale.\n\n"
        "Rules:\n"
        "- Always include a clear RATIONALE.\n"
        "- Never claim certainty; use language like 'I suggest', 'consider', 'based on the data'.\n"
        "- Do NOT hallucinate columns or values not present in the dataset context.\n"
        "- If project context is missing, ask the user to provide it in the Project context box.\n"
        "- Answer in the same language as the user's message.\n\n"
        "Output format (exact):\n"
        "ANSWER:\n<your answer>\n\nRATIONALE:\n<your rationale>\n"
    )


def _feedback_system_prompt() -> str:
    return (
        "You are Lab Notebook AI. Provide concise, action-oriented feedback about the dataset quality and EDA. "
        "Do NOT hallucinate columns or values not present in the dataset context. "
        "Use plain language for investigator-facing feedback."
    )


def build_agent_answer(
    message: str,
    rag_active: bool,
    dataset_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    settings = get_settings()

    # Safety: check both message and user-provided context.
    enforce_content_safety(message, settings.content_safety_blocklist)
    if user_context:
        enforce_content_safety(user_context, settings.content_safety_blocklist)

    session_id = (session_id or "").strip()
    if session_id and user_context is not None:
        set_user_context(session_id, user_context)
    effective_context = get_effective_context(session_id, user_context)

    citations: List[Dict[str, str]] = []
    side_panel = {
        "investigator": {"title": "What other researchers are doing", "items": []},
        "technical": {"rag_active": rag_active, "sources": []},
    }

    if rag_active:
        rag = search_scientific_context(query=message, top_k=5, use_local_mock=settings.use_local_mock)
        citations = rag.papers[:3]
        side_panel["investigator"]["items"] = [
            {"type": "paper", "title": p["title"], "source": p.get("source", "")} for p in rag.papers
        ] + [
            {"type": "dataset", "title": d["title"], "source": d.get("source", "")} for d in rag.datasets
        ]
        side_panel["technical"]["sources"] = rag.papers + rag.datasets

    cfg = _foundry_cfg()

    # Local mock or missing config: keep UX stable.
    # Force mock in pytest to avoid external calls.
    if os.getenv("PYTEST_CURRENT_TEST") or settings.use_local_mock or not is_configured(cfg):
        if not dataset_id:
            answer = (
                "Please upload a dataset first (Experiment/My data). Then ask me about variables, missingness, "
                "quarantine reasons, or what to analyze next."
            )
        else:
            ctx = dataset_context_json(dataset_id, max_chars=2500)
            answer = (
                "I can help analyze your uploaded dataset and suggest next EDA steps. "
                "For example: ask 'which columns have the most missingness?' or 'what variables look like IDs?'.\n\n"
                f"Dataset context (preview):\n{ctx}"
            )

        if not effective_context:
            answer += "\n\nBefore we go deeper, please add your Project context (goal, assay/protocol, units, and outcome definition)."

        rationale = (
            "Rationale: keeping project context + dataset context explicit reduces mistakes and lets each suggestion be traceable."
        )

        return {"answer": answer, "rationale": rationale, "citations": citations, "side_panel": side_panel}

    # LLM path
    dataset_ctx = ""
    if dataset_id:
        dataset_ctx = dataset_context_json(dataset_id, max_chars=9000)

    system_messages: List[Dict[str, str]] = [
        {"role": "system", "content": _base_system_prompt()},
        {
            "role": "system",
            "content": "Persistent project context (always in scope):\n" + (effective_context or "NOT PROVIDED"),
        },
    ]
    if dataset_ctx:
        system_messages.append({"role": "system", "content": f"Dataset context (dataset_id={dataset_id}):\n{dataset_ctx}"})

    history = get_history(session_id)
    messages = system_messages + history + [{"role": "user", "content": message}]

    text = chat_completions(cfg, messages=messages, temperature=0.2, max_tokens=700)
    answer, rationale = _parse_answer(text)

    if session_id:
        append_history(session_id, "user", message, max_messages=settings.llm_max_history_messages)
        append_history(session_id, "assistant", f"ANSWER:\n{answer}\n\nRATIONALE:\n{rationale}", max_messages=settings.llm_max_history_messages)

    return {"answer": answer, "rationale": rationale, "citations": citations, "side_panel": side_panel}


def build_data_feedback(
    dataset_id: str,
    session_id: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    settings = get_settings()

    # Safety
    if user_context:
        enforce_content_safety(user_context, settings.content_safety_blocklist)

    session_id = (session_id or "").strip()
    if session_id and user_context is not None:
        set_user_context(session_id, user_context)
    effective_context = get_effective_context(session_id, user_context)

    cfg = _foundry_cfg()

    if os.getenv("PYTEST_CURRENT_TEST") or settings.use_local_mock or not is_configured(cfg):
        return {
            "investigator": {
                "summary": "AI feedback is running in local mock mode.",
                "bullets": [
                    "Review quarantine reasons (outliers/all-null rows) before modeling.",
                    "Run EDA and inspect missingness + distributions for key variables.",
                    "If you have an outcome column, check outcome balance (counts vs %).",
                ],
                "note": "Add Project context to get more domain-specific suggestions.",
            },
            "technical": {
                "mode": "mock",
                "dataset_id": dataset_id,
                "next": ["/api/part1/eda", "/api/part1/quarantine/sample"],
            },
        }

    dataset_ctx = dataset_context_json(dataset_id, max_chars=9000)

    prompt = (
        "Provide feedback on this dataset's quality and EDA. Include quarantine insights and practical next steps. "
        "Keep it concise and action-oriented. Do not hallucinate.\n\n"
        "Return JSON exactly with keys: investigator, technical.\n"
        "investigator: {summary: str, bullets: [str], warnings: [str]}\n"
        "technical: {risks: [str], checks: [str], notes: str}\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _feedback_system_prompt()},
        {"role": "system", "content": "Persistent project context (always in scope):\n" + (effective_context or "NOT PROVIDED")},
        {"role": "system", "content": f"Dataset context (dataset_id={dataset_id}):\n{dataset_ctx}"},
        {"role": "user", "content": prompt},
    ]

    text = chat_completions(cfg, messages=messages, temperature=0.2, max_tokens=650)

    # Best-effort JSON parse
    try:
        js = json.loads(text)
        inv = js.get("investigator") or {}
        tech = js.get("technical") or {}
        if isinstance(inv, dict) and isinstance(tech, dict):
            return {"investigator": inv, "technical": tech}
    except Exception:
        pass

    return {
        "investigator": {
            "summary": "AI feedback (unstructured)",
            "bullets": [text[:800]],
            "warnings": [],
        },
        "technical": {"raw": text},
    }


def build_search_insight(
    query: str,
    papers: List[Dict[str, Any]],
    datasets: List[Dict[str, Any]],
    dataset_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    settings = get_settings()

    enforce_content_safety(query, settings.content_safety_blocklist)
    if user_context:
        enforce_content_safety(user_context, settings.content_safety_blocklist)

    session_id = (session_id or "").strip()
    if session_id and user_context is not None:
        set_user_context(session_id, user_context)
    effective_context = get_effective_context(session_id, user_context)

    # Compact evidence for prompt (include stable IDs so the insight can reference items)
    p = [
        {
            "id": f"P{i+1}",
            "title": x.get("title"),
            "year": x.get("year"),
            "venue": x.get("venue"),
            "url": x.get("url"),
        }
        for i, x in enumerate((papers or [])[:6])
    ]
    d = [
        {"id": f"D{i+1}", "title": x.get("title"), "url": x.get("url"), "source": x.get("source")}
        for i, x in enumerate((datasets or [])[:6])
    ]

    cfg = _foundry_cfg()
    if os.getenv("PYTEST_CURRENT_TEST") or settings.use_local_mock or not is_configured(cfg):
        refs: List[str] = []
        if p:
            refs.append(f"[{p[0]['id']}]")
        if d:
            refs.append(f"[{d[0]['id']}]")
        ref_txt = (" " + " ".join(refs)) if refs else ""
        return {
            "insight": (
                "These related papers/datasets can help you validate variable definitions, typical covariates, and expected ranges."
                + ref_txt
                + " Use them to refine your project context and decide what metadata (batch, units, protocol version, timepoint) "
                "should be added to your dataset for stronger analysis."
            ),
            "rationale": "Rationale: literature-backed context reduces ambiguity and improves traceability.",
        }

    dataset_ctx = ""
    if dataset_id:
        dataset_ctx = dataset_context_json(dataset_id, max_chars=8000)

    prompt = (
        "You are helping a researcher interpret 'related research' search results. "
        "Write a concise AI comment explaining: (1) how these papers/datasets can add context, "
        "(2) what metadata fields could enrich the user's dataset, and (3) 2-3 next actions.\n\n"
        "Important: explicitly reference specific items by their IDs in brackets (e.g., [P2], [D1]). "
        "If possible, mention at least two IDs so the user can map your advice to the list.\n\n"
        "Return in English (always).\n"
        "Output format (exact):\n"
        "ANSWER:\n<comment>\n\nRATIONALE:\n<why this helps>\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _base_system_prompt()},
        {"role": "system", "content": "Language: English."},
        {"role": "system", "content": "Persistent project context (always in scope):\n" + (effective_context or "NOT PROVIDED")},
        {"role": "system", "content": "Search query:\n" + query},
        {"role": "system", "content": "Top papers (metadata, with IDs):\n" + json.dumps(p, ensure_ascii=False, indent=2)},
        {"role": "system", "content": "Top datasets (metadata, with IDs):\n" + json.dumps(d, ensure_ascii=False, indent=2)},
    ]
    if dataset_ctx:
        messages.append({"role": "system", "content": f"Dataset context (dataset_id={dataset_id}):\n{dataset_ctx}"})
    messages.append({"role": "user", "content": prompt})

    text = chat_completions(cfg, messages=messages, temperature=0.2, max_tokens=500)
    ans, rat = _parse_answer(text)
    return {"insight": ans, "rationale": rat}


def explain_content_safety_error(err: ContentSafetyError) -> Dict[str, str]:
    return {
        "answer": "I can't process that request due to safety policy.",
        "rationale": "Rationale: the request matches blocked high-risk categories (RAI).",
        "error": str(err),
    }
