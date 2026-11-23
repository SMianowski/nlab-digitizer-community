"""Offline math support for the Qt Markdown help browser.

The help reference is also published as the project wiki, so it uses the same
TeX delimiters the wiki renders: ``$$ ... $$`` for display math and ``$...$``
inside a sentence. Qt's Markdown reader knows neither; formulas are swapped for
placeholder words before parsing and replaced by Matplotlib images afterwards.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl
from PySide6.QtGui import (
    QFontMetrics,
    QImage,
    QPalette,
    QTextCharFormat,
    QTextDocument,
    QTextFormat,
    QTextImageFormat,
)
from PySide6.QtWidgets import QTextBrowser

if TYPE_CHECKING:
    from matplotlib.figure import Figure

log = logging.getLogger(__name__)
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)$")
_CODE_SPAN = re.compile(r"(`+).+?\1")
# The pandoc/GitHub rule: no space just inside either $, and no digit right
# after the closing one, so prices such as "$5 and $10" stay text. $$ is display.
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?=\S)([^$\n]*?\S)\$(?![$\d])")


@dataclass(frozen=True)
class _Formula:
    token: str
    expression: str
    inline: bool


def _replace_inline_math(line: str, placeholder: Callable[[str], str]) -> str:
    """Swap $...$ spans for placeholders, leaving `code spans` untouched."""

    def swap(text: str) -> str:
        return _INLINE_MATH.sub(lambda match: placeholder(match[1]), text)

    parts: list[str] = []
    last = 0
    for code in _CODE_SPAN.finditer(line):
        parts.extend((swap(line[last:code.start()]), code[0]))
        last = code.end()
    parts.append(swap(line[last:]))
    return "".join(parts)


def _math_placeholders(markdown: str) -> tuple[str, list[_Formula]]:
    """Replace display and inline math outside code, leaving other Markdown intact."""
    lines = markdown.splitlines()
    output: list[str] = []
    formulas: list[_Formula] = []
    prefix = "NLABMATH" + sha256(markdown.encode()).hexdigest()[:16]

    def placeholder(expression: str, *, inline: bool) -> str:
        token = f"{prefix}{'I' if inline else 'X'}{len(formulas)}END"
        formulas.append(_Formula(token, expression, inline))
        return token

    fence = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        marker = _FENCE.fullmatch(line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                if not marker[2].strip():
                    fence = ""
        elif marker:
            fence = marker[1]
        else:
            stripped = line.strip()
            expression = ""
            end = index
            if line.startswith(("    ", "\t")):
                output.append(line)
                index += 1
                continue
            if stripped == "$$":
                end = next(
                    (i for i in range(index + 1, len(lines)) if lines[i].strip() == "$$"),
                    index,
                )
                expression = " ".join(lines[index + 1:end]).strip()
            elif stripped.startswith("$$") and stripped.endswith("$$"):
                expression = stripped[2:-2].strip()
            if expression:
                output.extend(("", placeholder(expression, inline=False), ""))
                index = end + 1
                continue
            line = _replace_inline_math(
                line, lambda found: placeholder(found, inline=True)
            )
        output.append(line)
        index += 1
    return "\n".join(output), formulas


@lru_cache(maxsize=128)
def _formula_image(expression: str, color: str, font_size: float, scale: float) -> QImage:
    # Import only when a math topic is opened. Explicitly disabling usetex
    # keeps rendering offline even if another Matplotlib user enables TeX.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(1, 0.5), dpi=96 * scale)
    FigureCanvasAgg(figure)
    figure.text(
        0, 0, f"${expression}$", color=color, fontsize=font_size,
        fontfamily="DejaVu Sans", math_fontfamily="dejavusans", usetex=False,
    )
    return _png_image(figure, bbox_inches="tight", pad_inches=0.04)


@lru_cache(maxsize=128)
def _inline_formula_image(
    expression: str, color: str, font_size: float, scale: float, half_x_height: float,
) -> QImage:
    """An inline formula whose baseline lands on the text baseline in Qt.

    Qt lays out an AlignMiddle inline image with ascent (H + x/2) / 2 and
    descent (H - x/2) / 2, where x is the text's x-height (QTextDocumentLayout::
    resizeInlineObject). Transparent rows above or below the formula make the
    part under its baseline exactly that descent.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser

    dpi = 96 * scale
    font = FontProperties(family="DejaVu Sans", size=font_size, math_fontfamily="dejavusans")
    source = f"${expression}$"
    width, height, depth, *_ = MathTextParser("path").parse(source, dpi=dpi, prop=font)
    margin = scale  # room for antialiasing at the glyph edges, in image pixels
    above = height - depth + margin
    below = depth + margin
    skew = above - below - half_x_height * scale
    if skew > 0:
        below += skew
    else:
        above -= skew
    figure_width = width + 2 * margin
    figure_height = above + below
    figure = Figure(figsize=(figure_width / dpi, figure_height / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    figure.text(
        margin / figure_width, below / figure_height, source, color=color,
        fontproperties=font, usetex=False,  # vertical alignment: baseline
    )
    return _png_image(figure)


def _png_image(figure: Figure, **save_options: object) -> QImage:
    stream = BytesIO()
    try:
        figure.savefig(
            stream, format="png", dpi=figure.dpi, transparent=True, **save_options,
        )
    finally:
        figure.clear()
    image = QImage.fromData(stream.getvalue())
    if image.isNull():
        raise RuntimeError("Could not render the help formula")
    return image


def set_help_markdown(browser: QTextBrowser, markdown: str) -> None:
    """Render Markdown plus TeX math, retaining the TeX as image alt text."""
    converted, formulas = _math_placeholders(markdown)
    browser.setMarkdown(converted)
    if not formulas:
        return
    document = browser.document()
    color = browser.palette().color(QPalette.ColorRole.Text).name()
    point_size = browser.font().pointSizeF()
    if point_size <= 0:
        point_size = max(browser.font().pixelSize(), 12) * 72 / 96
    display_size = max(12.0, point_size + 2)
    half_x_height = QFontMetrics(document.defaultFont()).xHeight() / 2
    scale = max(2.0, browser.devicePixelRatioF())
    for formula in formulas:
        cursor = document.find(formula.token)
        if cursor.isNull():
            continue
        expression = formula.expression
        try:
            if formula.inline:
                image = _inline_formula_image(
                    expression, color, point_size, scale, half_x_height,
                )
            else:
                image = _formula_image(expression, color, display_size, scale)
        except (OSError, RuntimeError, ValueError):
            log.warning("Could not render a help formula: %s", expression, exc_info=True)
            cursor.insertText(f"${expression}$" if formula.inline else f"$$ {expression} $$")
            continue
        digest = sha256(
            f"{expression}|{formula.inline}|{color}|{point_size}|{scale}|{half_x_height}"
            .encode()
        ).hexdigest()
        url = QUrl(f"nlab-math:{digest}")
        document.addResource(QTextDocument.ResourceType.ImageResource, url, image)
        image_format = QTextImageFormat()
        image_format.setName(url.toString())
        image_format.setWidth(image.width() / scale)
        image_format.setHeight(image.height() / scale)
        image_format.setProperty(QTextFormat.Property.ImageAltText, expression)
        image_format.setToolTip(expression)
        if formula.inline:
            image_format.setVerticalAlignment(QTextCharFormat.VerticalAlignment.AlignMiddle)
        cursor.insertImage(image_format)
