from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    user_context: str = ""
    history: List[Dict[str, str]] = field(default_factory=list)  # OpenAI-style: {role, content}


_SESSIONS: Dict[str, SessionState] = {}


def get_session(session_id: str) -> SessionState:
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionState(session_id=session_id)
    st = _SESSIONS[session_id]
    st.updated_at = time.time()
    return st


def set_user_context(session_id: str, context: Optional[str]) -> None:
    if not session_id:
        return
    st = get_session(session_id)
    if context is None:
        return
    st.user_context = (context or "").strip()
    st.updated_at = time.time()


def append_history(session_id: str, role: str, content: str, max_messages: int = 24) -> None:
    if not session_id:
        return
    st = get_session(session_id)
    st.history.append({"role": role, "content": content})
    if len(st.history) > max_messages:
        st.history = st.history[-max_messages:]
    st.updated_at = time.time()


def get_history(session_id: str) -> List[Dict[str, str]]:
    if not session_id or session_id not in _SESSIONS:
        return []
    return list(_SESSIONS[session_id].history)


def get_effective_context(session_id: str, inline_context: Optional[str]) -> str:
    inline = (inline_context or "").strip()
    if inline:
        return inline
    if session_id and session_id in _SESSIONS:
        return (_SESSIONS[session_id].user_context or "").strip()
    return ""
