from __future__ import annotations

import pytest

from choravox import report


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


# -- extensions --
# -- extensions --
