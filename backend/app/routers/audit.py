from fastapi import APIRouter

from app.state import get_state

router = APIRouter(prefix="/api", tags=["audit"])


@router.get("/audit-logs")
def get_audit_logs():
    state = get_state()
    return state.audit_logger.get_audit_trail()
