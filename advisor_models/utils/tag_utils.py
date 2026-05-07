"""Utilities for parsing structured <diagnosis><advice> advisor output.

Advisors are trained to produce responses in this format:
    <diagnosis>...</diagnosis><advice>...</advice>

These helpers extract each section and are shared across all env implementations.
"""

import re


def extract_diagnosis(text: str) -> str:
    """Return the content of the first <diagnosis>...</diagnosis> block.

    Returns an empty string if the tag is absent or empty.
    """
    m = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def extract_advice(text: str) -> str:
    """Return the content of the first <advice>...</advice> block.

    Falls back to the full text if the tag is absent so the student still
    receives something useful during early training when the advisor has not
    yet learned the structured format.
    """
    m = re.search(r"<advice>(.*?)</advice>", text, re.DOTALL)
    if not m:
        return text.strip()
    return m.group(1).strip()
