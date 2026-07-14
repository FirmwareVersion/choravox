from __future__ import annotations

import pytest

from choravox import report
from choravox.metrics import DiarizationError
from choravox.types import Segment


def test_svg_document_is_accessible_and_deterministic() -> None:
    first = report.svg_document("<circle/>", width=320, height=180, title="Synthetic chart")
    second = report.svg_document("<circle/>", width=320, height=180, title="Synthetic chart")
    assert first == second
    assert first.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert '<title id="title">Synthetic chart</title>' in first
    assert first.endswith("</svg>")


def test_report_text_escapes_hostile_markup_and_quotes() -> None:
    escaped = report.escape_text('<script data-x="1">& bad</script>')
    assert escaped == "&lt;script data-x=&quot;1&quot;&gt;&amp; bad&lt;/script&gt;"
    document = report.svg_document("", width=1, height=1, title="<unsafe>")
    assert "<unsafe>" not in document and "&lt;unsafe&gt;" in document


def test_svg_document_validates_canvas_geometry() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        report.svg_document("", width=0, height=10, title="chart")


def test_svg_wrapper_has_a_stable_golden_layout() -> None:
    assert report.svg_document("<g/>", width=10, height=20, title="T", description="D") == (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="20" '
        'viewBox="0 0 10 20" role="img" aria-labelledby="title desc">'
        '<title id="title">T</title><desc id="desc">D</desc>'
        '<rect width="100%" height="100%" fill="#fff"/><g/></svg>'
    )


def test_svg_description_is_escaped_independently() -> None:
    document = report.svg_document(
        "", width=10, height=10, title="title", description="measured <offline> & online"
    )
    assert "measured &lt;offline&gt; &amp; online" in document


def test_svg_body_is_treated_as_internal_markup() -> None:
    document = report.svg_document('<path d="M0 0"/>', width=10, height=10, title="chart")
    assert '<path d="M0 0"/>' in document


def test_finite_helper_rejects_negative_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="score"):
        report._finite_nonnegative(-1.0, "score")
    with pytest.raises(ValueError, match="score"):
        report._finite_nonnegative(float("nan"), "score")


def test_speaker_timeline_renders_named_rows_and_scaled_segments() -> None:
    svg = report.speaker_timeline_svg(
        (Segment(0.0, 1.0, "a"), Segment(1.0, 2.0, "b")), duration_s=2.0, width=400
    )
    assert ">a</text>" in svg and ">b</text>" in svg
    assert svg.count("<rect x=") == 2
    assert "0.000-1.000s" in svg and "1.000-2.000s" in svg


def test_speaker_timeline_parallel_rows_expose_overlap() -> None:
    svg = report.speaker_timeline_svg(
        (Segment(0.0, 1.0, "a"), Segment(0.5, 1.0, "b")), duration_s=1.0
    )
    assert 'y="30"' in svg and 'y="58"' in svg
    assert "parallel rows indicate overlap" in svg


def test_timeline_is_invariant_to_segment_input_order() -> None:
    segments = (Segment(0.5, 1.0, "b"), Segment(0.0, 0.5, "a"))
    assert report.speaker_timeline_svg(segments, duration_s=1.0) == report.speaker_timeline_svg(
        tuple(reversed(segments)), duration_s=1.0
    )


def test_timeline_escapes_hostile_speaker_labels() -> None:
    svg = report.speaker_timeline_svg((Segment(0.0, 1.0, "<host&ile>"),), duration_s=1.0)
    assert "<host&ile>" not in svg
    assert "&lt;host&amp;ile&gt;" in svg


def test_empty_timeline_retains_a_valid_canvas() -> None:
    svg = report.speaker_timeline_svg((), duration_s=1.0)
    assert "Speaker timeline" in svg and 'height="68"' in svg


def test_timeline_rejects_out_of_bounds_segments_and_small_width() -> None:
    with pytest.raises(ValueError, match="invalid"):
        report.speaker_timeline_svg((Segment(0.0, 2.0, "a"),), duration_s=1.0)
    with pytest.raises(ValueError, match="invalid"):
        report.speaker_timeline_svg((), duration_s=1.0, width=100)


def test_der_chart_contains_each_measured_component() -> None:
    result = DiarizationError(2, 3, 1, 20, 18)
    svg = report.der_decomposition_svg(result)
    assert all(label in svg for label in ("Miss", "False alarm", "Confusion"))
    assert "Measured synthetic DER 0.300000" in svg
    assert svg.count("<rect x=") == 3


def test_der_chart_handles_perfect_result_without_zero_scale() -> None:
    svg = report.der_decomposition_svg(DiarizationError(0, 0, 0, 10, 10))
    assert "Measured synthetic DER 0.000000" in svg
    assert svg.count('width="0.0"') == 3


def test_largest_der_component_fills_available_plot_width() -> None:
    svg = report.der_decomposition_svg(DiarizationError(2, 4, 1, 10, 10), width=500)
    assert 'width="320.0"' in svg


def test_der_component_widths_preserve_count_ratios() -> None:
    svg = report.der_decomposition_svg(DiarizationError(1, 2, 4, 10, 10), width=500)
    assert 'width="80.0"' in svg
    assert 'width="160.0"' in svg
    assert 'width="320.0"' in svg


def test_der_chart_output_is_byte_deterministic() -> None:
    result = DiarizationError(1, 2, 3, 20, 20)
    assert report.der_decomposition_svg(result) == report.der_decomposition_svg(result)


def test_der_chart_rejects_unreadable_canvas() -> None:
    with pytest.raises(ValueError, match="canvas"):
        report.der_decomposition_svg(DiarizationError(0, 0, 0, 1, 1), width=100)


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
