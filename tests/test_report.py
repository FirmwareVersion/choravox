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


# -- extensions --
