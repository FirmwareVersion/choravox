from __future__ import annotations

import pytest

from choravox import report
from choravox.metrics import DiarizationError
from choravox.streaming import LatencyQualityPoint
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


def test_metric_curve_sorts_points_and_labels_axes() -> None:
    svg = report.metric_curve_svg(
        ((10.0, 0.1), (0.0, 0.5), (5.0, 0.2)),
        title="SNR slices",
        x_label="SNR (dB)",
        y_label="DER",
    )
    assert "SNR slices" in svg and "SNR (dB)" in svg and ">DER</text>" in svg
    assert svg.count("<circle") == 3
    assert svg.index("0.000, 0.500000") < svg.index("10.000, 0.100000")


def test_constant_metric_curve_remains_finite() -> None:
    svg = report.metric_curve_svg(
        ((0.0, 0.2), (1.0, 0.2)), title="constant", x_label="x", y_label="y"
    )
    assert "nan" not in svg.lower() and "inf" not in svg.lower()


def test_single_point_curve_is_centered_on_valid_axes() -> None:
    svg = report.metric_curve_svg(
        ((3.0, 0.25),), title="single", x_label="condition", y_label="score"
    )
    assert svg.count("<circle") == 1
    assert "3.000, 0.250000" in svg


def test_metric_curve_is_invariant_to_input_order() -> None:
    points = ((0.0, 0.4), (1.0, 0.2), (2.0, 0.1))
    first = report.metric_curve_svg(points, title="curve", x_label="x", y_label="y")
    second = report.metric_curve_svg(
        tuple(reversed(points)), title="curve", x_label="x", y_label="y"
    )
    assert first == second


def test_metric_curve_escapes_axis_labels_and_title() -> None:
    svg = report.metric_curve_svg(((0.0, 0.0),), title="<curve>", x_label="x&x", y_label='y"y')
    assert "&lt;curve&gt;" in svg and "x&amp;x" in svg and "y&quot;y" in svg


def test_metric_curve_rejects_empty_duplicate_and_nonfinite_points() -> None:
    with pytest.raises(ValueError, match="requires"):
        report.metric_curve_svg((), title="x", x_label="x", y_label="y")
    with pytest.raises(ValueError, match="unique"):
        report.metric_curve_svg(((0.0, 1.0), (0.0, 2.0)), title="x", x_label="x", y_label="y")
    with pytest.raises(ValueError, match="finite"):
        report.metric_curve_svg(((0.0, float("nan")),), title="x", x_label="x", y_label="y")


def test_latency_quality_chart_contains_der_and_cpwer_series() -> None:
    points = (
        LatencyQualityPoint(0.1, 0.2, 0.3, 0.05),
        LatencyQualityPoint(0.2, 0.1, 0.2, 0.1),
    )
    markup = report.latency_quality_svg(points)
    assert markup.count("<svg") == 2
    assert "Latency-quality tradeoff (DER)" in markup
    assert "Latency-quality tradeoff (cpWER)" in markup
    assert 'data-source="synthetic"' in markup


def test_latency_quality_chart_rejects_unmeasured_empty_curve() -> None:
    with pytest.raises(ValueError, match="measured"):
        report.latency_quality_svg(())


def test_latency_quality_chart_is_input_order_invariant() -> None:
    points = (
        LatencyQualityPoint(0.1, 0.3, 0.4, 0.05),
        LatencyQualityPoint(0.2, 0.2, 0.3, 0.1),
    )
    assert report.latency_quality_svg(points) == report.latency_quality_svg(tuple(reversed(points)))


def test_latency_quality_chart_preserves_measured_tooltips() -> None:
    point = LatencyQualityPoint(0.125, 0.05, 0.1, 0.08)
    markup = report.latency_quality_svg((point,))
    assert "0.125, 0.050000" in markup
    assert "0.125, 0.100000" in markup


def test_latency_quality_chart_uses_requested_width_for_both_series() -> None:
    point = LatencyQualityPoint(0.1, 0.1, 0.1, 0.05)
    markup = report.latency_quality_svg((point,), width=640)
    assert markup.count('width="640"') == 2


def test_latency_quality_markup_is_self_contained() -> None:
    point = LatencyQualityPoint(0.1, 0.1, 0.1, 0.05)
    markup = report.latency_quality_svg((point,))
    assert "http://" in markup
    assert "<script" not in markup and "href=" not in markup


def test_run_metadata_hashes_json_compatible_config() -> None:
    metadata = report.RunMetadata.from_config(
        package_version="0.1.0",
        seed=7,
        config={"sample_rate": 16000, "speakers": 2},
        python_version="3.13.7",
    )
    assert len(metadata.config_hash) == 64
    assert report.metadata_table(metadata)[1] == ("seed", "7")
    assert report.metadata_table(metadata)[-1] == ("data", "deterministic synthetic meeting")


def test_run_metadata_rejects_bad_digest_and_seed() -> None:
    with pytest.raises(ValueError, match="metadata"):
        report.RunMetadata("0.1.0", -1, "0" * 64, "3.13")
    with pytest.raises(ValueError, match="metadata"):
        report.RunMetadata("0.1.0", 1, "not-a-digest", "3.13")


def test_config_hash_is_key_order_invariant() -> None:
    first = report.RunMetadata.from_config(
        package_version="0.1.0", seed=1, config={"a": 1, "b": 2}, python_version="3.13"
    )
    second = report.RunMetadata.from_config(
        package_version="0.1.0", seed=1, config={"b": 2, "a": 1}, python_version="3.13"
    )
    assert first.config_hash == second.config_hash


def test_config_hash_changes_with_behavioral_configuration() -> None:
    first = report.RunMetadata.from_config(
        package_version="0.1.0", seed=1, config={"speakers": 2}, python_version="3.13"
    )
    second = report.RunMetadata.from_config(
        package_version="0.1.0", seed=1, config={"speakers": 3}, python_version="3.13"
    )
    assert first.config_hash != second.config_hash


def test_metadata_table_has_stable_row_order() -> None:
    metadata = report.RunMetadata("0.1.0", 3, "a" * 64, "3.12")
    assert [name for name, _ in report.metadata_table(metadata)] == [
        "choravox",
        "seed",
        "config SHA-256",
        "Python",
        "data",
    ]


def test_metadata_config_must_be_json_serializable() -> None:
    with pytest.raises(TypeError):
        report.RunMetadata.from_config(
            package_version="0.1.0", seed=1, config={1, 2}, python_version="3.13"
        )


def test_limitations_state_synthetic_only_scope() -> None:
    facts = report.CoverageFacts(True, True, True, False)
    limitations = report.report_limitations(facts)
    assert limitations[0].startswith("All audio") and "synthetic" in limitations[0]
    assert any("No real corpus" in item for item in limitations)
    assert any("PyTorch" in item for item in limitations)
    assert not any("Overlap-aware" in item for item in limitations)


def test_unmeasured_paths_add_specific_limitations() -> None:
    facts = report.CoverageFacts(False, False, False, False)
    limitations = report.report_limitations(facts)
    assert any("Overlap-aware DER" in item for item in limitations)
    assert any("Online latency" in item for item in limitations)
    assert any("planted timestamps" in item for item in limitations)


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
