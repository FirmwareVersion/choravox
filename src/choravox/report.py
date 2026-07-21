"""Deterministic SVG, Markdown, and HTML reports from measured pipeline output."""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass

from choravox.hashing import content_hash
from choravox.metrics import DiarizationError
from choravox.streaming import LatencyQualityPoint
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


def der_decomposition_svg(result: DiarizationError, *, width: int = 520, height: int = 190) -> str:
    """Render measured miss, false-alarm, and confusion speaker-frame counts."""
    if width < 240 or height < 140:
        raise ValueError("DER chart canvas is too small")
    components = (
        ("Miss", result.miss, "#d95f59"),
        ("False alarm", result.false_alarm, "#d18b2c"),
        ("Confusion", result.confusion, "#9b6fc2"),
    )
    scale = max(1, max(value for _, value, _ in components))
    plot_width = width - 180
    elements = []
    for index, (label, value, color) in enumerate(components):
        y = 36 + index * 42
        bar_width = round(value / scale * plot_width, 3)
        elements.append(
            f'<text x="10" y="{y + 15}" font-size="12">{label}</text>'
            f'<rect x="110" y="{y}" width="{bar_width}" height="20" fill="{color}"/>'
            f'<text x="{120 + bar_width}" y="{y + 15}" font-size="12">{value}</text>'
        )
    return svg_document(
        "".join(elements),
        width=width,
        height=height,
        title="DER decomposition",
        description=(
            f"Measured synthetic DER {result.rate:.6f} over "
            f"{result.reference_speaker_frames} speaker frames."
        ),
    )


def metric_curve_svg(
    points: tuple[tuple[float, float], ...],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 560,
    height: int = 260,
) -> str:
    """Render a deterministic single-series curve from measured numeric points."""
    if not points or width < 260 or height < 180:
        raise ValueError("metric curve requires points and a readable canvas")
    if any(not all(math.isfinite(value) for value in point) for point in points):
        raise ValueError("metric curve points must be finite")
    ordered = tuple(sorted(points))
    if len({x for x, _ in ordered}) != len(ordered):
        raise ValueError("metric curve x values must be unique")
    left, top, plot_width, plot_height = 65, 30, width - 95, height - 80
    xs = [point[0] for point in ordered]
    ys = [point[1] for point in ordered]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    coordinates = [
        (
            round(left + (x - x_min) / x_span * plot_width, 3),
            round(top + (y_max - y) / y_span * plot_height, 3),
        )
        for x, y in ordered
    ]
    polyline = " ".join(f"{x},{y}" for x, y in coordinates)
    circles = "".join(
        (
            f'<circle cx="{x}" cy="{y}" r="3" fill="#3568c0">'
            f"<title>{source_x:.3f}, {source_y:.6f}</title></circle>"
        )
        for (x, y), (source_x, source_y) in zip(coordinates, ordered, strict=True)
    )
    bottom = top + plot_height
    middle_x = left + plot_width / 2
    middle_y = top + plot_height / 2
    body = (
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_width}" '
        f'y2="{bottom}" stroke="#333"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#3568c0" '
        f'stroke-width="2"/>{circles}'
        f'<text x="{middle_x}" y="{height - 12}" text-anchor="middle">'
        f"{escape_text(x_label)}</text>"
        f'<text x="14" y="{middle_y}" transform="rotate(-90 14 {middle_y})" '
        f'text-anchor="middle">{escape_text(y_label)}</text>'
    )
    return svg_document(body, width=width, height=height, title=title)


def latency_quality_svg(points: tuple[LatencyQualityPoint, ...], *, width: int = 600) -> str:
    """Plot measured online DER and cpWER against total latency as two SVG series."""
    if not points:
        raise ValueError("latency-quality chart requires measured points")
    ordered = tuple(sorted(points))
    der = metric_curve_svg(
        tuple((point.latency_s, point.der) for point in ordered),
        title="Latency-quality tradeoff (DER)",
        x_label="Latency (s)",
        y_label="DER",
        width=width,
    )
    cpwer = metric_curve_svg(
        tuple((point.latency_s, point.cpwer) for point in ordered),
        title="Latency-quality tradeoff (cpWER)",
        x_label="Latency (s)",
        y_label="cpWER",
        width=width,
    )
    return (
        '<div class="latency-quality" data-source="synthetic">'
        f"<figure>{der}<figcaption>DER</figcaption></figure>"
        f"<figure>{cpwer}<figcaption>cpWER</figcaption></figure></div>"
    )


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Reproducibility facts embedded in every synthesized meeting report."""

    package_version: str
    seed: int
    config_hash: str
    python_version: str

    def __post_init__(self) -> None:
        if (
            not self.package_version
            or self.seed < 0
            or re.fullmatch(r"[0-9a-f]{64}", self.config_hash) is None
            or not self.python_version
        ):
            raise ValueError("report run metadata is invalid")

    @classmethod
    def from_config(
        cls, *, package_version: str, seed: int, config: object, python_version: str
    ) -> RunMetadata:
        """Hash a JSON-compatible configuration at the report production point."""
        return cls(package_version, seed, content_hash(config), python_version)


def metadata_table(metadata: RunMetadata) -> tuple[tuple[str, str], ...]:
    """Expose stable display rows for Markdown and HTML renderers."""
    return (
        ("choravox", metadata.package_version),
        ("seed", str(metadata.seed)),
        ("config SHA-256", metadata.config_hash),
        ("Python", metadata.python_version),
        ("data", "deterministic synthetic meeting"),
    )


@dataclass(frozen=True, slots=True)
class CoverageFacts:
    """Facts about which synthetic evaluation paths actually ran."""

    planted_timestamps: bool
    overlap_scored: bool
    streaming_scored: bool
    torch_models_run: bool
    real_corpus_run: bool = False


def report_limitations(facts: CoverageFacts) -> tuple[str, ...]:
    """Generate honest limitations directly from recorded coverage facts."""
    limitations = [
        "All audio, speakers, transcripts, timestamps, and reported metrics are synthetic."
    ]
    if not facts.real_corpus_run:
        limitations.append("No real corpus or external speech recording was evaluated.")
    if not facts.torch_models_run:
        limitations.append("Optional trainable PyTorch models were not evaluated in this run.")
    if not facts.overlap_scored:
        limitations.append("Overlap-aware DER was not measured in this run.")
    if not facts.streaming_scored:
        limitations.append("Online latency and online/offline quality gaps were not measured.")
    if not facts.planted_timestamps:
        limitations.append("Forced-alignment output was not checked against planted timestamps.")
    return tuple(limitations)


# -- extensions --
