"""Deterministic PII detection and redaction with Australian formats.

Kinds: email, phone_au (mobiles, landlines with or without area-code brackets, +61 international),
tfn (8 or 9 digits, check-digit validated), medicare (10 or 11 digits, check-digit validated),
credit_card (13 to 19 digits, Luhn validated) and abn (11 digits, weighting validated). Check digits
cut false positives on ordinary numbers; a 9-digit invoice number passes the TFN check one time in
eleven, and a phone number is never mistaken for a card.

`redact()` replaces each finding with `[REDACTED:kind]` and returns the findings with their spans in
the original text. `detect_only()` returns the findings alone. Both are pure functions of their input.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict

DEFAULT_KINDS: tuple[str, ...] = ("email", "phone_au", "tfn", "medicare", "credit_card", "abn")


class Finding(TypedDict):
    """One detected value: its kind and (start, end) span in the original text."""

    kind: str
    span: tuple[int, int]


@dataclass(frozen=True)
class _Candidate:
    kind: str
    start: int
    end: int


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn_ok(value: str) -> bool:
    """Luhn checksum over a digit string (payment cards)."""
    digits = _digits(value)
    if len(digits) < 13:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _tfn_ok(value: str) -> bool:
    """ATO tax file number check: weighted sum divisible by 11 (9-digit and legacy 8-digit forms)."""
    digits = _digits(value)
    if len(digits) == 9:
        weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    elif len(digits) == 8:
        weights = (10, 7, 8, 4, 6, 3, 5, 1)
    else:
        return False
    if len(set(digits)) == 1:
        return False
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


def _medicare_ok(value: str) -> bool:
    """Medicare card check: first digit 2 to 6, weighted sum of digits 1 to 8 mod 10 equals digit 9."""
    digits = _digits(value)
    if len(digits) not in (10, 11) or digits[0] not in "23456":
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9)
    check = sum(int(d) * w for d, w in zip(digits[:8], weights, strict=True)) % 10
    return check == int(digits[8])


def _abn_ok(value: str) -> bool:
    """ABN check: subtract 1 from the first digit, weighted sum divisible by 89."""
    digits = _digits(value)
    if len(digits) != 11:
        return False
    weights = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    values = [int(digits[0]) - 1] + [int(d) for d in digits[1:]]
    if values[0] < 0:
        return False
    return sum(v * w for v, w in zip(values, weights, strict=True)) % 89 == 0


def _always(_: str) -> bool:
    return True


# kind -> (pattern, validator). Lookarounds keep a match from starting or ending inside a longer digit run.
# Digit groups may be separated by one whitespace character (a space, a tab, or the line break PDF and
# DOCX extraction puts inside a wrapped number) or a hyphen. Email addresses use Unicode word classes so
# internationalised local parts and domains are caught, and a match never starts inside a word.
_SPECS: dict[str, tuple[re.Pattern[str], Callable[[str], bool]]] = {
    "email": (
        re.compile(r"(?<![\w.%+-])[\w.%+-]+@[\w-]+(?:\.[\w-]+)+(?![\w-])"),
        _always,
    ),
    "phone_au": (
        re.compile(
            r"(?<![\d+])(?:"
            r"\+61[\s-]?(?:\(0\)[\s-]?)?[2-478](?:[\s-]?\d){8}"  # +61 4xx xxx xxx, +61 2 xxxx xxxx
            r"|\(0[2-478]\)[\s-]?\d(?:[\s-]?\d){7}"  # (02) xxxx xxxx
            r"|0[2-478](?:[\s-]?\d){8}"  # 04xx xxx xxx, 02 xxxx xxxx
            r")(?!\d)"
        ),
        _always,
    ),
    "tfn": (re.compile(r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?\d{2,3}(?!\d)"), _tfn_ok),
    "medicare": (re.compile(r"(?<!\d)[2-6]\d{3}[\s-]?\d{5}[\s-]?\d(?:[\s-]?\d)?(?!\d)"), _medicare_ok),
    "credit_card": (re.compile(r"(?<!\d)\d(?:[\s-]?\d){12,18}(?!\d)"), _luhn_ok),
    "abn": (re.compile(r"(?<!\d)\d{2}[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)"), _abn_ok),
}

# Tie-break for candidates with the same start and length: longer, more specific formats first.
_PRECEDENCE: tuple[str, ...] = ("email", "credit_card", "abn", "medicare", "tfn", "phone_au")


def _candidates(text: str, kinds: Sequence[str]) -> list[_Candidate]:
    found: list[_Candidate] = []
    for kind in kinds:
        if kind not in _SPECS:
            raise ValueError(f"unknown PII kind {kind!r}; choose from {', '.join(_SPECS)}")
        pattern, valid = _SPECS[kind]
        for m in pattern.finditer(text):
            if valid(m.group(0)):
                found.append(_Candidate(kind, m.start(), m.end()))
    return found


def _resolve(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep non-overlapping candidates: earliest start, then longest, then precedence."""
    ordered = sorted(candidates, key=lambda c: (c.start, -(c.end - c.start), _PRECEDENCE.index(c.kind)))
    chosen: list[_Candidate] = []
    last_end = -1
    for cand in ordered:
        if cand.start >= last_end:
            chosen.append(cand)
            last_end = cand.end
    return chosen


def detect_only(text: str, kinds: Sequence[str] = DEFAULT_KINDS) -> list[Finding]:
    """Return findings for the requested kinds without changing the text."""
    return [{"kind": c.kind, "span": (c.start, c.end)} for c in _resolve(_candidates(text, kinds))]


def redact(text: str, kinds: Sequence[str] = DEFAULT_KINDS) -> tuple[str, list[Finding]]:
    """Replace each finding with `[REDACTED:kind]` and return (redacted_text, findings).

    Spans in the findings refer to the original text.
    """
    findings = detect_only(text, kinds)
    pieces: list[str] = []
    cursor = 0
    for f in findings:
        start, end = f["span"]
        pieces.append(text[cursor:start])
        pieces.append(f"[REDACTED:{f['kind']}]")
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), findings
