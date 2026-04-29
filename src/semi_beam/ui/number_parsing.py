from __future__ import annotations

from typing import Optional


def normalize_decimal_text(text: str, allow_negative: bool = True) -> str:
    raw = str(text if text is not None else "")
    out: list[str] = []
    has_decimal = False
    has_digit_before_sign = False
    sign_added = False

    for ch in raw:
        if ch.isdigit():
            out.append(ch)
            has_digit_before_sign = True
            continue
        if ch in ".,":
            if not has_decimal:
                out.append(".")
                has_decimal = True
            continue
        if ch == "-" and allow_negative and not sign_added and not has_digit_before_sign and not out:
            out.append("-")
            sign_added = True

    return "".join(out)


def try_parse_user_float(text: str, *, allow_negative: bool = True) -> Optional[float]:
    normalized = normalize_decimal_text(text, allow_negative=allow_negative).strip()
    if normalized in {"", "-", ".", "-."}:
        return None
    try:
        return float(normalized)
    except Exception:
        return None


def parse_user_float(text: str, *, allow_negative: bool = True) -> float:
    value = try_parse_user_float(text, allow_negative=allow_negative)
    if value is None:
        raise ValueError(f"No se pudo interpretar como numero: {text!r}")
    return float(value)


def normalize_line_edit_text(line_edit, *, allow_negative: bool = True) -> None:
    before = line_edit.text() or ""
    after = normalize_decimal_text(before, allow_negative=allow_negative)
    if after == before:
        return

    old_cursor = int(line_edit.cursorPosition())
    normalized_before_cursor = normalize_decimal_text(before[:old_cursor], allow_negative=allow_negative)
    new_cursor = min(len(after), len(normalized_before_cursor))

    blocked = line_edit.blockSignals(True)
    try:
        line_edit.setText(after)
        line_edit.setCursorPosition(new_cursor)
    finally:
        line_edit.blockSignals(blocked)
