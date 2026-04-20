#!/usr/bin/env python3
"""Convert SHARING_PLAN.md to a clean PDF using reportlab."""

import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Preformatted, HRFlowable,
)


def md_inline(text: str) -> str:
    """Convert inline markdown (bold, italic, code) to reportlab XML."""
    text = text.replace("&", "&amp;")
    # Escape < and > but not our own tags we'll insert
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', text)
    return text


def build_pdf(md_text: str, output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MDTitle", parent=styles["Title"], fontSize=20, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    )
    h2_style = ParagraphStyle(
        "MDH2", parent=styles["Heading2"], fontSize=14, spaceBefore=16,
        spaceAfter=6, textColor=HexColor("#16213e"),
    )
    h3_style = ParagraphStyle(
        "MDH3", parent=styles["Heading3"], fontSize=12, spaceBefore=12,
        spaceAfter=4, textColor=HexColor("#0f3460"),
    )
    body_style = ParagraphStyle(
        "MDBody", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=8,
    )
    code_style = ParagraphStyle(
        "MDCode", fontName="Courier", fontSize=8, leading=10,
        leftIndent=12, spaceAfter=8, spaceBefore=4,
        backColor=HexColor("#f4f4f4"),
    )
    quote_style = ParagraphStyle(
        "MDQuote", parent=body_style, fontSize=9.5, leading=13,
        leftIndent=24, rightIndent=12, spaceAfter=8,
        textColor=HexColor("#444444"), borderPadding=4,
        backColor=HexColor("#f9f9f9"),
    )
    list_style = ParagraphStyle(
        "MDList", parent=body_style, fontSize=10, leading=14,
        leftIndent=24, spaceAfter=3,
    )
    list_cont_style = ParagraphStyle(
        "MDListCont", parent=body_style, fontSize=10, leading=14,
        leftIndent=36, spaceAfter=3,
    )

    story = []
    lines = md_text.split("\n")
    i = 0
    # Track ordered list numbering across gaps
    ol_counter = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Blank line - reset OL counter if next non-blank isn't a numbered item
        if not stripped:
            # Look ahead to see if OL continues
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and not re.match(r"^\d+\.\s", lines[j].strip()):
                # Check if it's indented continuation of a list item
                if j < len(lines) and not lines[j].startswith("   "):
                    ol_counter = 0
            i += 1
            continue

        # Horizontal rule
        if stripped == "---":
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cccccc")))
            story.append(Spacer(1, 6))
            ol_counter = 0
            i += 1
            continue

        # H1
        if stripped.startswith("# ") and not stripped.startswith("## "):
            story.append(Paragraph(md_inline(stripped[2:]), title_style))
            story.append(Spacer(1, 4))
            ol_counter = 0
            i += 1
            continue

        # H2
        if stripped.startswith("## ") and not stripped.startswith("### "):
            story.append(Paragraph(md_inline(stripped[3:]), h2_style))
            ol_counter = 0
            i += 1
            continue

        # H3
        if stripped.startswith("### "):
            story.append(Paragraph(md_inline(stripped[4:]), h3_style))
            ol_counter = 0
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            code_text = "\n".join(code_lines)
            # Truncate very long code blocks
            cl = code_text.split("\n")
            if len(cl) > 20:
                cl = cl[:18] + ["    ..."]
                code_text = "\n".join(cl)
            code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Preformatted(code_text, code_style))
            continue

        # Table
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            table_rows = []
            while i < len(lines) and "|" in lines[i]:
                if "---" not in lines[i]:
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    table_rows.append(cells)
                i += 1

            if table_rows:
                cell_style = ParagraphStyle("Cell", parent=body_style, fontSize=8.5, leading=11, spaceAfter=0)
                header_cell = ParagraphStyle("HCell", parent=cell_style, fontName="Helvetica-Bold")

                data = []
                for ri, row in enumerate(table_rows):
                    s = header_cell if ri == 0 else cell_style
                    data.append([Paragraph(md_inline(c), s) for c in row])

                ncols = len(data[0])
                col_w = doc.width / ncols
                t = Table(data, colWidths=[col_w] * ncols)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8eaf6")),
                    ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
            continue

        # Blockquote (possibly indented under a list item)
        if re.match(r"^\s*> ", line) or stripped.startswith(">"):
            quote_parts = []
            while i < len(lines) and (re.match(r"^\s*> ", lines[i]) or lines[i].strip().startswith(">")):
                raw = re.sub(r"^\s*>\s?", "", lines[i])
                quote_parts.append(raw.strip())
                i += 1
            # Join with actual line breaks for display
            quote_text = "<br/>".join([md_inline(p) for p in quote_parts])
            story.append(Paragraph(quote_text, quote_style))
            continue

        # Ordered list item
        m = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m:
            ol_counter = int(m.group(1))
            item_text = m.group(2)
            # Gather continuation lines (indented lines that follow)
            i += 1
            while i < len(lines) and lines[i].startswith("   ") and not re.match(r"^\s*>\s", lines[i]) and not lines[i].strip().startswith("```"):
                item_text += " " + lines[i].strip()
                i += 1
            story.append(Paragraph(f"{ol_counter}. {md_inline(item_text)}", list_style))

            # Check for sub-elements (blockquotes, code) under this list item
            while i < len(lines):
                if re.match(r"^\s*> ", lines[i]) or (lines[i].strip().startswith(">") and lines[i].startswith("   ")):
                    # Indented blockquote under list item
                    quote_parts = []
                    while i < len(lines) and (re.match(r"^\s*> ", lines[i]) or lines[i].strip().startswith(">")):
                        raw = re.sub(r"^\s*>\s?", "", lines[i])
                        quote_parts.append(raw.strip())
                        i += 1
                    quote_text = "<br/>".join([md_inline(p) for p in quote_parts])
                    nested_quote = ParagraphStyle(
                        "NestedQuote", parent=quote_style, leftIndent=36,
                    )
                    story.append(Paragraph(quote_text, nested_quote))
                elif lines[i].strip().startswith("```"):
                    # Indented code block under list item
                    code_lines = []
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        code_lines.append(lines[i])
                        i += 1
                    i += 1
                    ct = "\n".join(code_lines)
                    ct = ct.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    story.append(Preformatted(ct, code_style))
                elif lines[i].strip() == "":
                    i += 1
                    # Check if next content is still part of list context
                    continue
                else:
                    break
            continue

        # Unordered list
        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            for item in items:
                story.append(Paragraph(f"\u2022  {md_inline(item)}", list_style))
            story.append(Spacer(1, 4))
            ol_counter = 0
            continue

        # Regular paragraph - collect continuation lines
        if stripped:
            para_lines = []
            while i < len(lines) and lines[i].strip() \
                    and not lines[i].startswith("#") \
                    and not lines[i].strip().startswith("```") \
                    and not lines[i].strip() == "---" \
                    and not ("|" in lines[i] and i + 1 < len(lines) and "---" in lines[i + 1]) \
                    and not re.match(r"^\d+\.\s", lines[i].strip()) \
                    and not lines[i].strip().startswith("- ") \
                    and not re.match(r"^\s*> ", lines[i]):
                para_lines.append(lines[i].strip())
                i += 1
            story.append(Paragraph(md_inline(" ".join(para_lines)), body_style))
            ol_counter = 0
            continue

        i += 1

    doc.build(story)


if __name__ == "__main__":
    md_path = Path("/Users/liamsandy/Documents/Legal/CivilRightsSummarizedAI/SHARING_PLAN.md")
    out_path = md_path.parent / "SHARING_PLAN.pdf"
    md_text = md_path.read_text()
    build_pdf(md_text, str(out_path))
    print(f"PDF saved to {out_path}")
