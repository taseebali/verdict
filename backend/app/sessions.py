"""Per-visitor in-memory sessions keyed by an HttpOnly cookie.

Nothing here touches disk: a restart or idle expiry simply forgets the data.
"""
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException, Request, Response

COOKIE_NAME = "verdict_sid"
NO_DATASET = "No dataset loaded — load the demo or upload a CSV first."
NO_MODEL = "No model trained yet — pick an outcome first."


@dataclass
class NewScores:
    """A second file scored with the trained model (no known outcomes)."""

    name: str
    df: pd.DataFrame
    proba: np.ndarray


@dataclass
class Session:
    id: str
    last_seen: float
    df: Optional[pd.DataFrame] = None
    dataset_name: Optional[str] = None
    model: Any = None
    new_scores: Optional[NewScores] = None
    reasons_cache: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_model(self) -> None:
        """Forget everything derived from the current dataset."""
        self.model = None
        self.new_scores = None
        self.reasons_cache = {}


class SessionStore:
    """LRU-capped, idle-expiring map of session id -> Session."""

    def __init__(self, max_sessions: int = 30, ttl_seconds: float = 3600,
                 clock: Callable[[], float] = time.monotonic):
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._max = max_sessions
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str]) -> tuple[Session, bool]:
        now = self._clock()
        with self._lock:
            self._expire(now)
            session = self._sessions.get(session_id) if session_id else None
            if session is not None:
                session.last_seen = now
                self._sessions.move_to_end(session.id)
                return session, False
            session = Session(id=secrets.token_urlsafe(32), last_seen=now)
            self._sessions[session.id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)
            return session, True

    def _expire(self, now: float) -> None:
        stale = [sid for sid, s in self._sessions.items() if now - s.last_seen > self._ttl]
        for sid in stale:
            del self._sessions[sid]

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)


store = SessionStore()


def get_session(request: Request, response: Response) -> Session:
    """FastAPI dependency: the caller's session, created (and cookied) on first use."""
    session, created = store.get_or_create(request.cookies.get(COOKIE_NAME))
    if created:
        set_session_cookie(request, response, session.id)
    return session


def _is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def set_session_cookie(request: Request, response: Response, session_id: str) -> None:
    """Behind https (HF Spaces serves the app in a cross-site iframe) the cookie
    must be SameSite=None; Secure; Partitioned (CHIPS) or browsers won't send it.
    Starlette 0.41's set_cookie has no `partitioned`, so the header is built here."""
    if _is_https(request):
        response.headers.append(
            "set-cookie",
            f"{COOKIE_NAME}={session_id}; HttpOnly; Path=/; SameSite=None; Secure; Partitioned",
        )
    else:
        response.set_cookie(COOKIE_NAME, session_id, httponly=True, samesite="lax", path="/")


def require_dataset(session: Session) -> pd.DataFrame:
    if session.df is None:
        raise HTTPException(status_code=404, detail=NO_DATASET)
    return session.df


def require_model(session: Session):
    require_dataset(session)
    if session.model is None:
        raise HTTPException(status_code=404, detail=NO_MODEL)
    return session.model
