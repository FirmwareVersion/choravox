"""Deterministic SVG, Markdown, and HTML reports from measured pipeline output."""

from __future__ import annotations

import html
import math

from choravox.types import Segment


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


_COLORS = ("#3568c0", "#d95f59", "#3c9d6f", "#9b6fc2", "#d18b2c", "#3b8ea5")


def speaker_timeline_svg(
    segments: tuple[Segment, ...], *, duration_s: float, width: int = 720
) -> str:
    """Render measured single- and multi-speaker activity as a timeline ribbon."""
    _finite_nonnegative(duration_s, "timeline duration")
    if duration_s == 0.0 or width < 160 or any(segment.end > duration_s for segment in segments):
        raise ValueError("timeline duration, width, and segment bounds are invalid")
    speakers = tuple(sorted({segment.speaker for segment in segments}))
    row_height = 28
    height = 40 + row_height * max(1, len(speakers))
    plot_left, plot_width = 90, width - 110
    rows = {speaker: index for index, speaker in enumerate(speakers)}
    elements = [
        f'<line x1="{plot_left}" y1="20" x2="{plot_left + plot_width}" y2="20" stroke="#333"/>'
    ]
    for speaker, row in rows.items():
        y = 30 + row * row_height
        elements.append(f'<text x="8" y="{y + 14}" font-size="12">{escape_text(speaker)}</text>')
    for segment in sorted(segments):
        x = plot_left + round(segment.start / duration_s * plot_width, 3)
        rectangle_width = round((segment.end - segment.start) / duration_s * plot_width, 3)
        y = 30 + rows[segment.speaker] * row_height
        color = _COLORS[rows[segment.speaker] % len(_COLORS)]
        elements.append(
            f'<rect x="{x}" y="{y}" width="{rectangle_width}" height="18" '
            f'fill="{color}" rx="2"><title>{escape_text(segment.speaker)} '
            f"{segment.start:.3f}-{segment.end:.3f}s</title></rect>"
        )
    return svg_document(
        "".join(elements),
        width=width,
        height=height,
        title="Speaker timeline",
        description="Synthetic meeting speaker activity; parallel rows indicate overlap.",
    )


# -- extensions --
