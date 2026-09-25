"""Guards against path traversal in the SPA static-file fallback (main.py's
serve_spa catch-all route). A request escaping FRONTEND_DIST must never
return the contents of an arbitrary file on disk.
"""
import re

from app.main import FRONTEND_DIST

_HAS_FRONTEND_BUILD = FRONTEND_DIST.exists()


def _get_requirements_txt_content() -> str:
    repo_root = FRONTEND_DIST.parent.parent
    req_file = repo_root / "requirements.txt"
    return req_file.read_text() if req_file.exists() else ""


def _looks_like_index_html(text: str) -> bool:
    return bool(re.search(r"<div id=\"root\"", text)) or "<!doctype html" in text.lower()


def test_traversal_does_not_leak_arbitrary_file(client):
    if not _HAS_FRONTEND_BUILD:
        return  # nothing to serve without a built frontend; route isn't registered

    response = client.get("/../../requirements.txt")
    requirements_content = _get_requirements_txt_content()

    if requirements_content:
        assert requirements_content not in response.text

    # It should have fallen through to the SPA's index.html rather than
    # leaking a file outside FRONTEND_DIST.
    assert response.status_code == 200
    assert _looks_like_index_html(response.text)


def test_encoded_traversal_does_not_leak_arbitrary_file(client):
    if not _HAS_FRONTEND_BUILD:
        return

    response = client.get("/..%2F..%2Frequirements.txt")
    requirements_content = _get_requirements_txt_content()

    if requirements_content:
        assert requirements_content not in response.text
    assert response.status_code == 200
    assert _looks_like_index_html(response.text)


def test_api_typo_returns_404_not_html(client):
    if not _HAS_FRONTEND_BUILD:
        return

    response = client.get("/api/definitely-not-a-real-endpoint")
    assert response.status_code == 404
