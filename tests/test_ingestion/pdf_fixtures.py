"""PDF fixtures for ingestion tests."""

from __future__ import annotations

from pathlib import Path

import pymupdf


def write_empty_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


def write_image_only_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 80, 80), 1)
    pix.clear_with(80)
    page.insert_image(pymupdf.Rect(40, 40, 200, 200), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


def write_two_column_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    left = (
        "ALPHA COLUMN START\n"
        "This is the left column narrative about exit load.\n"
        "ALPHA_TOKEN uniquely marks the left column.\n"
        "ALPHA COLUMN END"
    )
    right = (
        "BETA COLUMN START\n"
        "This is the right column narrative about expense ratio.\n"
        "BETA_TOKEN uniquely marks the right column.\n"
        "BETA COLUMN END"
    )
    page.insert_textbox(pymupdf.Rect(40, 60, 280, 780), left, fontsize=11)
    page.insert_textbox(pymupdf.Rect(320, 60, 560, 780), right, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def write_factsheet_table_pdf(path: Path) -> Path:
    """Grid table with ruled lines so PyMuPDF table detection can fire."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 40), "Top 10 Equity Holdings (as on 30th June, 2026)", fontsize=12)

    headers = ["Company", "Industry", "% to NAV"]
    rows = [
        ["ICICI Bank Ltd.", "Banks", "9.18"],
        ["Axis Bank Ltd.", "Banks", "6.84"],
        ["HDFC Bank Ltd.", "Banks", "6.77"],
    ]
    col_x = [50, 250, 400, 520]
    row_y = [70, 100, 130, 160, 190]
    shape = page.new_shape()
    for y in row_y:
        shape.draw_line((col_x[0], y), (col_x[-1], y))
    for x in col_x:
        shape.draw_line((x, row_y[0]), (x, row_y[-1]))
    shape.finish(color=(0, 0, 0), width=0.6)
    shape.commit()

    def put(col: int, row: int, text: str) -> None:
        page.insert_text((col_x[col] + 6, row_y[row] + 18), text, fontsize=10)

    for i, h in enumerate(headers):
        put(i, 0, h)
    for r, row in enumerate(rows, start=1):
        for c, cell in enumerate(row):
            put(c, r, cell)

    page.insert_text((50, 230), "Investment Objective", fontsize=12)
    page.insert_textbox(
        pymupdf.Rect(50, 245, 520, 320),
        "To generate capital appreciation from a predominantly equity portfolio.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    return path


def write_sid_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "SCHEME INFORMATION DOCUMENT", fontsize=14)
    page.insert_text((72, 110), "PART II. INFORMATION ABOUT THE SCHEME", fontsize=12)
    page.insert_text((72, 140), "C. LOAD STRUCTURE", fontsize=12)
    body = (
        "Exit Load: 1.00% if Units are redeemed within 1 year from the date of allotment. "
        "No Exit Load is payable if Units are redeemed after 1 year from the date of allotment."
    )
    page.insert_textbox(pymupdf.Rect(72, 160, 520, 320), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def write_drhp_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "SECTION II - RISK FACTORS", fontsize=14)
    body = (
        "The top risk factor is concentration of revenue from jewellery manufacturing "
        "and retail operations in Gujarat. Investors should read all risk factors carefully."
    )
    page.insert_textbox(pymupdf.Rect(72, 110, 520, 400), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def write_drhp_toc_pdf(path: Path) -> Path:
    """TOC page with dotted leaders, then a real section heading."""
    doc = pymupdf.open()
    toc = doc.new_page()
    toc.insert_text((72, 72), "TABLE OF CONTENTS", fontsize=14)
    toc.insert_text((72, 110), "SECTION I – GENERAL ..................................................................... 1", fontsize=11)
    toc.insert_text(
        (72, 140),
        "SECTION VIII – DESCRIPTION OF EQUITY SHARES AND TERMS OF THE ARTICLES OF ASSOCIATION",
        fontsize=11,
    )
    toc.insert_text((72, 160), ".................................................................................... 426", fontsize=11)
    toc.insert_text(
        (72, 190),
        "SECTION IX: OTHER INFORMATION ................................................................. 438",
        fontsize=11,
    )
    body = doc.new_page()
    body.insert_text((72, 72), "SECTION II - RISK FACTORS", fontsize=14)
    body.insert_textbox(
        pymupdf.Rect(72, 110, 520, 400),
        "An investment in Equity Shares involves a high degree of risk. "
        "RISK_BODY_TOKEN marks the real section, not the table of contents.",
        fontsize=11,
    )
    doc.save(path)
    doc.close()
    return path


def write_sebi_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "CHAPTER IV", fontsize=14)
    page.insert_text((72, 110), "Regulation 18. Computation of net asset value", fontsize=12)
    body = (
        "18.1 Every mutual fund shall compute the NAV of each scheme by dividing "
        "the net assets by the number of units outstanding on the valuation date."
    )
    page.insert_textbox(pymupdf.Rect(72, 140, 520, 400), body, fontsize=11)
    doc.save(path)
    doc.close()
    return path
