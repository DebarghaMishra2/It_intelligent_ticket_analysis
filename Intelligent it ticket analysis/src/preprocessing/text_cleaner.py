
import re
import unicodedata
import math


SEARCH_FIELDS = [
    "title",
    "description",
    "symptoms",
    "application",
    "service",
]


# Remove only known anonymization and sensitive-data placeholders.
SENSITIVE_PLACEHOLDERS = [
    "LOCATION",
    "NAME",
    "COMPANY",
    "TICKET ID",
    "ADDRESS",
    "EMAIL",
    "USERNAME",
    "AWS_ACCESS_KEY",
    "PHONE",
    "PASSWORD",
    "URL",
    "IP_ADDRESS",
    "DRIVER_ID",
]

SENSITIVE_PATTERN = re.compile(
    r"\[(?:" + "|".join(
        re.escape(item) for item in SENSITIVE_PLACEHOLDERS
    ) + r")\]",
    flags=re.IGNORECASE
)


BOILERPLATE_PATTERNS = [
    r"\bnew support ticket received\b",
    r"\ba support ticket was forwarded to (?:your role|you)\b",
    r"\bsupport ticket received\b",
    r"\boriginal message\b",
]


def _safe_string(value):
    """Convert a value to a safe string."""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "<na>"}:
        return ""

    return text


def normalize_ticket_text(text):
    """
    Normalize ticket text for lexical and semantic retrieval.

    Known sensitive placeholders are removed.
    Other bracketed terms are preserved as searchable text.
    Original descriptions are never modified.
    """

    text = _safe_string(text)

    if not text:
        return ""

    # Unicode normalization.
    text = unicodedata.normalize("NFKC", text)

    # Lowercase for consistent searching.
    text = text.lower()

    # Remove standard ticket boilerplate.
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )

    # Remove only known sensitive placeholders.
    text = SENSITIVE_PATTERN.sub(" ", text)

    # Preserve useful bracketed terms by removing brackets,
    # not their contents.
    # Example: [Light House] -> light house
    # Example: [SERVER] -> server
    text = re.sub(r"[\[\]]", " ", text)

    # Remove ticket references such as T12345678.1.
    text = re.sub(
        r"\bt\d{8}\.\d+\b",
        " ",
        text
    )

    # Repair malformed phrases left by anonymization.
    text = re.sub(
        r"\b(unable\s+to)\s+from\b",
        r"\1",
        text
    )

    # Remove common email separator markers.
    text = re.sub(
        r"---+\s*",
        " ",
        text
    )

    # Normalize arrows into spaces while preserving their words.
    # Example: server up -> down becomes server up down.
    text = re.sub(
        r"\s*(?:-->|->)\s*",
        " ",
        text
    )

    # Remove standalone dash separators.
    # Preserve internal hyphens such as wi-fi and check_mk.
    text = re.sub(
        r"(?<!\w)-+(?!\w)",
        " ",
        text
    )

    # Remove stray leading hyphens from identifiers.
    # Example: -wsus01 becomes wsus01.
    text = re.sub(
        r"(?<!\w)-(?=\w)",
        "",
        text
    )

    # Normalize vertical separators.
    text = text.replace("|", " ")

    # Remove punctuation-only tokens.
    text = re.sub(
        r"(?<!\S)[.,;:!?]+(?!\S)",
        " ",
        text
    )

    # Remove empty angle brackets and similar artifacts.
    text = re.sub(r"<\s*>", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove leftover punctuation at boundaries.
    text = text.strip(" \t\n\r-–—|,;:")

    return text.strip()


def build_search_text(record):
    """
    Construct one searchable text field from available ticket fields.

    Supports dictionaries and pandas Series.
    Duplicate cleaned field values are included only once.
    """

    cleaned_parts = []

    for field in SEARCH_FIELDS:
        value = record.get(field, "")
        cleaned = normalize_ticket_text(value)

        if cleaned and cleaned not in cleaned_parts:
            cleaned_parts.append(cleaned)

    return " ".join(cleaned_parts)
