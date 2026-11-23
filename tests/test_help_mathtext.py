"""Offline Markdown equation rendering, display and inline, in the real Qt document."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import (
    QColor,
    QImage,
    QPalette,
    QTextCharFormat,
    QTextDocument,
    QTextFormat,
    QTextImageFormat,
)
from PySide6.QtWidgets import QTextBrowser
from pytestqt.qtbot import QtBot

from nlab.help import mathtext
from nlab.help.content import load_topics
from nlab.help.mathtext import set_help_markdown


def _images(browser: QTextBrowser) -> list[QTextImageFormat]:
    return [item.toImageFormat() for item in browser.document().allFormats()
            if item.isImageFormat()]


def test_display_math_renders_offline_with_fraction_and_greek_symbols(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    expression = r"\mathrm{FOM}=\frac{|\mu_2-\mu_1|}{\mathrm{FWHM}_1+\mathrm{FWHM}_2}"
    set_help_markdown(browser, "# Model\n\n$$\n" + expression + "\n$$\n\n[Quality](#fom-quality)")
    images = _images(browser)
    assert len(images) == 1
    image_format = images[0]
    assert image_format.property(QTextFormat.Property.ImageAltText) == expression
    assert image_format.toolTip() == expression
    image = browser.document().resource(
        QTextDocument.ResourceType.ImageResource, QUrl(image_format.name()),
    )
    assert isinstance(image, QImage) and not image.isNull()
    assert image.hasAlphaChannel()
    assert image.width() >= image_format.width() * 2
    assert image.height() >= image_format.height() * 2
    assert image_format.height() >= 20  # A real stacked fraction, not a broken-image icon.
    assert 'href="#fom-quality"' in browser.toHtml()
    assert "Model" in browser.toPlainText() and "Quality" in browser.toPlainText()


def test_code_fences_and_unclosed_math_remain_literal(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    markdown = "```text\n$$\nx^2\n$$\n```\n\n~~~~\n$$x^3$$\n~~~~\n\n$$\nunclosed"
    set_help_markdown(browser, markdown)
    assert not _images(browser)
    assert "$$" in browser.toPlainText() and "unclosed" in browser.toPlainText()


def test_single_line_math_and_cached_navigation(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    mathtext._formula_image.cache_clear()
    markdown = "Before\n\n$$y_0+G_1(x)+G_2(x)$$\n\nAfter"
    set_help_markdown(browser, markdown)
    assert len(_images(browser)) == 1
    set_help_markdown(browser, "Another topic")
    set_help_markdown(browser, markdown)
    assert len(_images(browser)) == 1
    assert mathtext._formula_image.cache_info().misses == 1
    assert mathtext._formula_image.cache_info().hits == 1


def test_invalid_math_keeps_help_readable(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    expression = r"\unknown_help_math_command{x}"
    set_help_markdown(browser, f"# Help\n\n$${expression}$$\n\nStill readable")
    assert not _images(browser)
    assert expression in browser.toPlainText()
    assert "Still readable" in browser.toPlainText()


def test_math_uses_the_help_foreground_color(qtbot: QtBot) -> None:
    mathtext._formula_image.cache_clear()
    light = QTextBrowser()
    dark = QTextBrowser()
    qtbot.addWidget(light)
    qtbot.addWidget(dark)
    for browser, foreground in ((light, "#111111"), (dark, "#eeeeee")):
        palette = browser.palette()
        palette.setColor(QPalette.ColorRole.Text, QColor(foreground))
        browser.setPalette(palette)
        set_help_markdown(browser, "$$x^2$$")
    brightness: list[float] = []
    for browser in (light, dark):
        image = browser.document().resource(
            QTextDocument.ResourceType.ImageResource, QUrl(_images(browser)[0].name()),
        )
        assert isinstance(image, QImage)
        opaque_pixels = [image.pixelColor(x, y).lightness() for y in range(image.height())
                         for x in range(image.width()) if image.pixelColor(x, y).alpha() > 200]
        assert opaque_pixels
        brightness.append(sum(opaque_pixels) / len(opaque_pixels))
    assert brightness[0] < 64 and brightness[1] > 192
    first, second = _images(light)[0], _images(dark)[0]
    assert first.name() != second.name()
    assert mathtext._formula_image.cache_info().misses == 2


def test_inline_math_renders_inside_its_sentence(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    set_help_markdown(
        browser,
        r"The window is $8\times 2^e$ ns and $1/R_{raw}=\lfloor R_{x}/4\rfloor$ is derived.",
    )
    images = _images(browser)
    assert [image.property(QTextFormat.Property.ImageAltText) for image in images] == [
        r"8\times 2^e", r"1/R_{raw}=\lfloor R_{x}/4\rfloor",
    ]
    for image_format in images:
        # Qt centres AlignMiddle objects on the text; the renderer pads for it.
        assert image_format.verticalAlignment() == QTextCharFormat.VerticalAlignment.AlignMiddle
        image = browser.document().resource(
            QTextDocument.ResourceType.ImageResource, QUrl(image_format.name()),
        )
        assert isinstance(image, QImage) and not image.isNull() and image.hasAlphaChannel()
    text = browser.toPlainText()
    assert "$" not in text and "_" not in text  # no raw TeX, no emphasis markers lost
    assert text.startswith("The window is ") and text.endswith(" is derived.")
    assert browser.document().blockCount() == 1  # still one paragraph


@pytest.mark.parametrize(
    "markdown",
    [
        "Prices $5 and $10 stay text.",
        r"An escaped \$x$ stays text.",
        "Code `$x$` stays code.",
        "Spaced $ x $ is not math.",
        "```\n$x$\n```",
        "    $x$ in an indented block",
    ],
)
def test_text_that_only_looks_like_inline_math_stays_literal(
    qtbot: QtBot, markdown: str,
) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    set_help_markdown(browser, markdown)
    assert not _images(browser)
    assert "$" in browser.toPlainText()


def test_invalid_inline_math_keeps_its_tex(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    set_help_markdown(browser, r"Before $\unknown_help_math_command{x}$ after")
    assert not _images(browser)
    assert browser.toPlainText() == r"Before $\unknown_help_math_command{x}$ after"


def test_every_help_topic_renders_its_math(qtbot: QtBot) -> None:
    browser = QTextBrowser()
    qtbot.addWidget(browser)
    for topic in load_topics().values():
        set_help_markdown(browser, f"# {topic.title}\n\n{topic.markdown}")
        assert "$" not in browser.toPlainText(), topic.key
