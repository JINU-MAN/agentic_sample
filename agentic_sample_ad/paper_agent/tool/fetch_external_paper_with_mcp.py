from __future__ import annotations

from ._core import fetch_external_paper_with_mcp as _fetch_external_paper_with_mcp_impl


def fetch_external_paper_with_mcp(
    reference: str = "",
    url: str = "",
    doi: str = "",
    arxiv_id: str = "",
    max_chars: int = 12000,
) -> str:
    """
    Fetch external paper metadata or a compact preview from URL / DOI / arXiv ID.
    """
    return _fetch_external_paper_with_mcp_impl(
        reference=reference,
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        max_chars=max_chars,
    )


__all__ = ["fetch_external_paper_with_mcp"]
