# /// script
# requires-python = ">=3.11"
# dependencies = ["reportlab"]
# ///
"""Generate deterministic synthetic pay stubs (no public-domain filled pay stubs
exist) plus one synthetic resume used as an irrelevant-document distractor.

Output: data/clean/pay_slip/*.pdf, data/clean/resume/*.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent

PAYSTUBS = [
    {
        "name": "paystub_rivera.pdf",
        "employer": "Cedar Ridge Logistics LLC",
        "employee": "Maria T. Rivera",
        "employee_id": "EMP-4417",
        "pay_period": "06/01/2026 - 06/15/2026",
        "pay_date": "06/20/2026",
        "gross": "$4,250.00",
        "federal_tax": "$510.00",
        "state_tax": "$170.00",
        "social_security": "$263.50",
        "medicare": "$61.63",
        "net": "$3,244.87",
        "ytd_gross": "$46,750.00",
    },
    {
        "name": "paystub_okafor.pdf",
        "employer": "Brightline Data Systems Inc",
        "employee": "Daniel C. Okafor",
        "employee_id": "EMP-2093",
        "pay_period": "06/01/2026 - 06/30/2026",
        "pay_date": "07/01/2026",
        "gross": "$7,500.00",
        "federal_tax": "$1,125.00",
        "state_tax": "$375.00",
        "social_security": "$465.00",
        "medicare": "$108.75",
        "net": "$5,426.25",
        "ytd_gross": "$45,000.00",
    },
    {
        "name": "paystub_lindqvist.pdf",
        "employer": "Harborview Medical Group",
        "employee": "Anna S. Lindqvist",
        "employee_id": "EMP-8850",
        "pay_period": "06/08/2026 - 06/21/2026",
        "pay_date": "06/26/2026",
        "gross": "$3,100.00",
        "federal_tax": "$310.00",
        "state_tax": "$124.00",
        "social_security": "$192.20",
        "medicare": "$44.95",
        "net": "$2,428.85",
        "ytd_gross": "$40,300.00",
    },
]


def draw_paystub(c: canvas.Canvas, s: dict) -> None:
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, h - 1 * inch, s["employer"])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, h - 1.3 * inch, "EARNINGS STATEMENT / PAY SLIP")

    c.setFont("Helvetica", 10)
    y = h - 1.8 * inch
    rows = [
        ("Employee Name", s["employee"]),
        ("Employee ID", s["employee_id"]),
        ("Pay Period", s["pay_period"]),
        ("Pay Date", s["pay_date"]),
        ("", ""),
        ("Gross Pay", s["gross"]),
        ("Federal Tax Withholding", s["federal_tax"]),
        ("State Tax Withholding", s["state_tax"]),
        ("Social Security", s["social_security"]),
        ("Medicare", s["medicare"]),
        ("Total Deductions", ""),
        ("Net Pay", s["net"]),
        ("", ""),
        ("YTD Gross Earnings", s["ytd_gross"]),
    ]
    for label, value in rows:
        if label:
            c.drawString(1 * inch, y, label)
            c.drawRightString(5.5 * inch, y, value)
        y -= 0.25 * inch

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1 * inch, 0.75 * inch,
                 "Synthetic document generated for the mortgage-doc-rag test corpus. Not a real record.")


RESUME_LINES = [
    ("Helvetica-Bold", 16, "Jordan K. Alvarez"),
    ("Helvetica", 10, "914 Willow Bend Dr, Austin, TX 78701 | jordan.alvarez@example.com"),
    ("Helvetica-Bold", 12, "Objective"),
    ("Helvetica", 10, "Operations analyst with six years of experience seeking a senior role."),
    ("Helvetica-Bold", 12, "Experience"),
    ("Helvetica", 10, "Senior Operations Analyst, Lakeshore Freight (2022-2026)"),
    ("Helvetica", 10, "Operations Analyst, Trailhead Retail Group (2020-2022)"),
    ("Helvetica-Bold", 12, "Education"),
    ("Helvetica", 10, "B.S. Business Administration, Texas State University, 2020"),
    ("Helvetica-Bold", 12, "Skills"),
    ("Helvetica", 10, "SQL, Excel, Tableau, process improvement, vendor management"),
    ("Helvetica-Bold", 12, "References"),
    ("Helvetica", 10, "Available upon request."),
]


def main() -> None:
    out_dir = ROOT / "data" / "clean" / "pay_slip"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in PAYSTUBS:
        c = canvas.Canvas(str(out_dir / s["name"]), pagesize=letter)
        draw_paystub(c, s)
        c.save()
        print(f"  ok  pay_slip/{s['name']}")

    resume_dir = ROOT / "data" / "clean" / "resume"
    resume_dir.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(resume_dir / "resume_alvarez.pdf"), pagesize=letter)
    w, h = letter
    y = h - 1 * inch
    for font, size, text in RESUME_LINES:
        c.setFont(font, size)
        c.drawString(1 * inch, y, text)
        y -= 0.3 * inch if size > 10 else 0.25 * inch
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1 * inch, 0.75 * inch,
                 "Synthetic document generated for the mortgage-doc-rag test corpus. Not a real record.")
    c.save()
    print("  ok  resume/resume_alvarez.pdf")


if __name__ == "__main__":
    main()
