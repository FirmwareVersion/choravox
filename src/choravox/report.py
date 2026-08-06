"""Deterministic SVG, Markdown, and HTML reports from measured pipeline output."""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from choravox.hashing import canonical_json, content_hash
from choravox.metrics import DiarizationError
from choravox.streaming import LatencyQualityPoint
from choravox.types import Segment, Transcript


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


@dataclass(frozen=True, slots=True)
class MeetingReport:
    title: str
    metadata: RunMetadata
    transcript: Transcript
    diarization: DiarizationError
    cpwer: float
    limitations: tuple[str, ...]
    charts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.title or self.cpwer < 0.0 or not math.isfinite(self.cpwer):
            raise ValueError("meeting report title and cpWER must be valid")
        if len({name for name, _ in self.charts}) != len(self.charts):
            raise ValueError("meeting report chart names must be unique")
        if any(not name or not _safe_inline_chart(chart) for name, chart in self.charts):
            raise ValueError("meeting report charts must contain safe inline markup")


def _safe_inline_chart(markup: str) -> bool:
    """Accept only chart markup emitted here while rejecting active content."""
    lowered = markup.casefold()
    allowed_root = markup.startswith(("<svg ", "<svg>")) or markup.startswith(
        '<div class="latency-quality" data-source="synthetic">'
    )
    forbidden = ("<script", "javascript:", " href=", " src=", " onload=", " onclick=")
    return allowed_root and not any(token in lowered for token in forbidden)


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|])")


def escape_markdown(value: str) -> str:
    """Escape report-controlled text in Markdown table and heading contexts."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", value).replace("\n", " ")


def markdown_report(meeting: MeetingReport) -> str:
    """Render one deterministic Markdown report from measured outputs."""
    lines = [
        f"# {escape_markdown(meeting.title)}",
        "",
        "> 本报告仅使用确定性合成会议与 planted ground truth; 指标不代表真实语料表现。",
        "",
        "## 运行元数据",
        "",
        "| 字段 | 值 |",
        "|---|---|",
    ]
    lines.extend(
        f"| {escape_markdown(name)} | {escape_markdown(value)} |"
        for name, value in metadata_table(meeting.metadata)
    )
    lines.extend(
        [
            "",
            "## 测量结果",
            "",
            f"- DER: {meeting.diarization.rate:.6f}",
            f"- cpWER: {meeting.cpwer:.6f}",
            f"- 参考 speaker-frames: {meeting.diarization.reference_speaker_frames}",
            "",
            "## 说话人归属转写",
            "",
            "| 开始 (s) | 结束 (s) | 说话人 | 词 | 置信度 |",
            "|---:|---:|---|---|---:|",
        ]
    )
    lines.extend(
        f"| {word.start:.3f} | {word.end:.3f} | {escape_markdown(word.speaker)} | "
        f"{escape_markdown(word.text)} | {word.confidence:.3f} |"
        for word in meeting.transcript.words
    )
    for name, chart in meeting.charts:
        lines.extend(("", f"## {escape_markdown(name)}", "", chart))
    lines.extend(("", "## 限制", ""))
    lines.extend(f"- {escape_markdown(item)}" for item in meeting.limitations)
    return "\n".join(lines) + "\n"


def html_report(meeting: MeetingReport) -> str:
    """Render a self-contained HTML report without scripts or external assets."""
    metadata_rows = "".join(
        f"<tr><th>{escape_text(name)}</th><td>{escape_text(value)}</td></tr>"
        for name, value in metadata_table(meeting.metadata)
    )
    transcript_rows = "".join(
        f"<tr><td>{word.start:.3f}</td><td>{word.end:.3f}</td>"
        f"<td>{escape_text(word.speaker)}</td><td>{escape_text(word.text)}</td>"
        f"<td>{word.confidence:.3f}</td></tr>"
        for word in meeting.transcript.words
    )
    charts = "".join(
        f"<section><h2>{escape_text(name)}</h2>{chart}</section>" for name, chart in meeting.charts
    )
    limitations = "".join(f"<li>{escape_text(item)}</li>" for item in meeting.limitations)
    style = (
        "body{font:15px system-ui,sans-serif;max-width:960px;margin:auto;padding:24px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th,td{border:1px solid #bbb;padding:6px;text-align:left}"
        "figure{margin:12px 0}svg{max-width:100%;height:auto}"
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{style}</style>'
        f"<title>{escape_text(meeting.title)}</title></head><body>"
        f"<h1>{escape_text(meeting.title)}</h1>"
        "<p><strong>范围:</strong> 确定性合成会议与 planted ground truth; 不代表真实语料表现。</p>"
        f"<h2>运行元数据</h2><table><tbody>{metadata_rows}</tbody></table>"
        f"<h2>测量结果</h2><ul><li>DER: {meeting.diarization.rate:.6f}</li>"
        f"<li>cpWER: {meeting.cpwer:.6f}</li></ul>"
        "<h2>说话人归属转写</h2><table><thead><tr><th>开始</th><th>结束</th>"
        f"<th>说话人</th><th>词</th><th>置信度</th></tr></thead><tbody>{transcript_rows}</tbody></table>"
        f"{charts}<h2>限制</h2><ul>{limitations}</ul></body></html>"
    )


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    markdown: str
    html: str
    digest: str


@dataclass(frozen=True, slots=True)
class ReportFile:
    name: str
    media_type: str
    digest: str
    size: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or "/" in self.name
            or not self.media_type
            or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None
            or self.size < 0
        ):
            raise ValueError("report file manifest entry is invalid")


@dataclass(frozen=True, slots=True)
class ReportManifest:
    report_digest: str
    files: tuple[ReportFile, ...]


REPORT_MANIFEST_SCHEMA = "choravox.report-manifest"
REPORT_MANIFEST_VERSION = 1


def report_manifest(artifacts: ReportArtifacts) -> ReportManifest:
    """Describe deterministic Markdown and HTML report payloads."""
    payloads = (
        ("report.md", "text/markdown; charset=utf-8", artifacts.markdown.encode()),
        ("report.html", "text/html; charset=utf-8", artifacts.html.encode()),
    )
    files = tuple(
        ReportFile(name, media_type, content_hash({"bytes": data.hex()}), len(data))
        for name, media_type, data in payloads
    )
    return ReportManifest(artifacts.digest, files)


def report_manifest_to_json(manifest: ReportManifest) -> str:
    """Encode a versioned manifest with a payload integrity digest."""
    payload = {
        "files": [
            {
                "digest": item.digest,
                "media_type": item.media_type,
                "name": item.name,
                "size": item.size,
            }
            for item in manifest.files
        ],
        "report_digest": manifest.report_digest,
    }
    document = {
        "header": {
            "digest": content_hash(payload),
            "schema": REPORT_MANIFEST_SCHEMA,
            "version": REPORT_MANIFEST_VERSION,
        },
        "payload": payload,
    }
    return canonical_json(document).decode() + "\n"


def report_manifest_from_json(text: str) -> ReportManifest:
    """Decode and authenticate a report manifest."""
    try:
        document = json.loads(text)
        header = document["header"]
        payload = document["payload"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("report manifest is malformed") from error
    if header.get("schema") != REPORT_MANIFEST_SCHEMA or header.get("version") != 1:
        raise ValueError("unsupported report manifest schema or version")
    if header.get("digest") != content_hash(payload):
        raise ValueError("report manifest digest mismatch")
    try:
        return ReportManifest(
            str(payload["report_digest"]),
            tuple(ReportFile(**item) for item in payload["files"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("report manifest payload is invalid") from error


def write_report_artifacts(
    directory: str | Path, artifacts: ReportArtifacts, *, overwrite: bool = False
) -> ReportManifest:
    """Write both report formats and their manifest to one output directory."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = tuple(destination / name for name in ("report.md", "report.html", "manifest.json"))
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("report output already exists")
    manifest = report_manifest(artifacts)
    payloads = (
        artifacts.markdown.encode(),
        artifacts.html.encode(),
        report_manifest_to_json(manifest).encode(),
    )
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(payload)
    return manifest


def synthesize_report(
    *,
    title: str,
    metadata: RunMetadata,
    transcript: Transcript,
    diarization: DiarizationError,
    cpwer: float,
    facts: CoverageFacts,
    charts: tuple[tuple[str, str], ...],
) -> ReportArtifacts:
    """Build both report formats and hash the exact deterministic inputs."""
    meeting = MeetingReport(
        title,
        metadata,
        transcript,
        diarization,
        cpwer,
        report_limitations(facts),
        charts,
    )
    markdown = markdown_report(meeting)
    document = html_report(meeting)
    digest = content_hash(
        {
            "charts": charts,
            "cpwer": cpwer,
            "der": {
                "confusion": diarization.confusion,
                "false_alarm": diarization.false_alarm,
                "miss": diarization.miss,
                "reference": diarization.reference_speaker_frames,
            },
            "metadata": metadata_table(metadata),
            "title": title,
            "transcript": [
                {
                    "confidence": word.confidence,
                    "end": word.end,
                    "speaker": word.speaker,
                    "start": word.start,
                    "text": word.text,
                }
                for word in transcript.words
            ],
        }
    )
    return ReportArtifacts(markdown, document, digest)


# -- extensions --
