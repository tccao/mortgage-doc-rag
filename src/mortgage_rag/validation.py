"""Deterministic regex validation over extracted text.

Acts as a rule-based backstop to the LLM: flags missing loan amounts, unmasked
SSNs, and cross-document inconsistencies that generation could otherwise paper
over. Also exposed to the agentic orchestrator as a tool.
"""

from __future__ import annotations

import re
from typing import Any

LOAN_AMOUNT_PATTERN = re.compile(r"\$([\d,]+\.\d{2})")
INTEREST_RATE_PATTERN = re.compile(r"(\d{1,2}\.\d{1,3})%")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DATE_PATTERN = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")


def validate_extracted_data(text: str, filename: str = "") -> tuple[dict[str, Any], list[str]]:
    """Regex logic checks on extracted text. Returns (data, issues)."""
    issues: list[str] = []
    data: dict[str, Any] = {}

    amounts = LOAN_AMOUNT_PATTERN.findall(text)
    if amounts:
        cleaned = [float(a.replace(",", "")) for a in amounts]
        # Heuristic: the largest dollar figure in a loan document is the loan amount.
        data["loan_amount"] = max(cleaned)
        data["all_amounts"] = cleaned
    else:
        issues.append("Missing loan amount")

    data["interest_rate"] = INTEREST_RATE_PATTERN.findall(text)

    if SSN_PATTERN.search(text):
        issues.append("Unmasked SSN detected - compliance risk")

    data["dates"] = DATE_PATTERN.findall(text)

    return data, issues


def cross_check_consistency(all_data: list[dict[str, Any]]) -> str:
    """Checks whether loan amounts agree across documents."""
    amounts = [
        entry["data"]["loan_amount"]
        for entry in all_data
        if "loan_amount" in entry.get("data", {})
    ]

    if not amounts:
        return "No loan amounts found to cross-check."

    if len(set(amounts)) > 1:
        return f"Mismatch in loan amounts across documents: {sorted(set(amounts))}"
    return f"Loan amounts consistent: ${amounts[0]:,.2f}"
