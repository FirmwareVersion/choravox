"""Deterministic SVG, Markdown, and HTML reports from measured pipeline output."""

from __future__ import annotations

import html
import math


def escape_text(value: str) -> str:
    """Escape hostile text for XML and HTML text or attribute contexts."""
    return html.escape(value, quote=True)


def svg_document(body: str, *, width: int, height: int, title: str, description: str = "") -> str:
    """Wrap trusted SVG elements in a deterministic accessible document."""
    if width < 1 or height < 1 or not title:
        raise ValueError("SVG dimensions and title must be valid")
    title_text = escape_text(title)
    description_text = escape_text(description)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{title_text}</title><desc id="desc">{description_text}</desc>'
        f'<rect width="100%" height="100%" fill="#fff"/>{body}</svg>'
    )


def _finite_nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


# -- extensions --
