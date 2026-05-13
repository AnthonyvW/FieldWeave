from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common.app_context import AppContext
from datetime import datetime

_TOKEN_OPEN  = "\uE000"  # sentinel for '{{'
_TOKEN_CLOSE = "\uE001"  # sentinel for '}}'
_FIELD_RE = re.compile(r"\{([^{}]+)\}")


def _nm_digit_width(max_mm: int) -> int:
    """Return the number of digits needed to represent max_mm converted to nm."""
    max_nm = max_mm * 1_000_000
    return max(1, len(str(abs(max_nm))))


def _trailing_zeros_from_step(step_nm: int) -> int:
    """
    Return how many trailing zero digits can be stripped from a zero-padded nm
    integer given the minimum step size.

    e.g. step_nm=40_000 -> every value is a multiple of 40_000, so the last 4
    digits are always '0000' and can be dropped, leaving the value divided by
    10_000 to display.  We find the largest power-of-10 that divides step_nm.
    """
    if step_nm <= 0:
        return 0
    count = 0
    n = step_nm
    while n % 10 == 0:
        count += 1
        n //= 10
    return count


def _format_nm(value_nm: int, full_width: int, strip_zeros: int) -> str:
    """
    Format value_nm as a zero-padded string of width (full_width - strip_zeros),
    representing (value_nm // 10**strip_zeros).
    """
    divisor = 10 ** strip_zeros
    display_value = value_nm // divisor
    display_width = max(1, full_width - strip_zeros)
    return f"{display_value:0{display_width}d}"


class ImageNameFormatter:
    """
    Safe image filename formatter.

    Recognised placeholders
    -----------------------
    {x} {y} {z}
        Stage positions in nanometres, zero-padded to the axis travel range and
        truncated by the minimum step size so unnecessary trailing zeros are
        omitted.  Axis maxima and step size are read from
        ``ctx.motion.settings`` at format time when no overrides are provided.

    {i}
        Image index.  Resolution order: explicit ``index`` arg >
        ``ctx.current_sample_index`` (if the attribute exists) > internal
        counter ``_index``.

    {d[:<strftime>]}
        Date/time from the system clock.  Custom format via ``{d:%Y%m%d_%H%M%S}``.
        Defaults to ``%Y%m%d`` when no format specifier is given.

    Unknown placeholders (e.g. ``{sample}``) are left intact.

    Axis padding / truncation
    -------------------------
    Padding width  : derived from ``max_x/y/z`` (mm) converted to nm.
    Trailing-zero strip : derived from ``step_size`` (nm) — digits that are
        always zero due to the minimum step are removed from both the value
        and the pad width, keeping filenames as short as possible while still
        being unambiguous.

    You can override either value via ``set_axis_maxes_nm`` and
    ``set_step_size_nm`` when no app context is available (e.g. in tests).
    """

    def __init__(
        self,
        *,
        pad_positions: bool = False,
        template: str | None = None,
        default_date_format: str = "%Y%m%d",
        start_index: int = 1,
    ) -> None:
        self.pad_positions = pad_positions
        self._template: str | None = template
        self._default_date_format = default_date_format
        self._index = int(start_index)

        # Manual overrides (used when app context is unavailable).
        self._axis_max_nm: dict[str, int | None] = {"x": None, "y": None, "z": None}
        self._step_size_nm: int | None = None

    # ---------------- Configuration ----------------

    def set_template(self, template: str) -> None:
        self._template = template

    def get_template(self) -> str | None:
        return self._template

    def set_index(self, value: int) -> None:
        self._index = int(value)

    def set_axis_maxes_nm(self, *, x: int | None = None, y: int | None = None, z: int | None = None) -> None:
        """Override axis travel maxima in nm (used when no app context is available)."""
        if x is not None:
            self._axis_max_nm["x"] = int(x)
        if y is not None:
            self._axis_max_nm["y"] = int(y)
        if z is not None:
            self._axis_max_nm["z"] = int(z)

    def set_step_size_nm(self, step_nm: int) -> None:
        """Override the minimum step size in nm (used when no app context is available)."""
        self._step_size_nm = int(step_nm)

    # ---------------- Internals ----------------

    def _motion_settings(self) -> Any | None:
        from common.app_context import get_app_context
        ctx: AppContext = get_app_context()
        return ctx.motion.settings if ctx is not None else None

    @staticmethod
    def _needed_fields(template: str) -> set[str]:
        needed: set[str] = set()
        for m in _FIELD_RE.finditer(template):
            key = m.group(1).strip().split(":", 1)[0].strip()
            needed.add(key)
        return needed

    def _axis_full_widths(self, needed: set[str]) -> dict[str, int]:
        """Digit widths for the full nm value (before stripping trailing zeros)."""
        widths: dict[str, int] = {"x": 1, "y": 1, "z": 1}
        ms = self._motion_settings()

        for axis in ("x", "y", "z"):
            if axis not in needed:
                continue
            override = self._axis_max_nm[axis]
            if override is not None:
                widths[axis] = max(1, len(str(abs(override))))
            elif ms is not None:
                max_mm = getattr(ms, f"max_{axis}", 1)
                widths[axis] = _nm_digit_width(int(max_mm))

        return widths

    def _strip_zeros(self) -> int:
        """Trailing zero digits to remove based on step size."""
        if self._step_size_nm is not None:
            return _trailing_zeros_from_step(self._step_size_nm)
        ms = self._motion_settings()
        if ms is not None:
            return _trailing_zeros_from_step(int(ms.step_size))
        return 0

    def _collect_values(
        self,
        needed: set[str],
        *,
        index: int | None,
        auto_increment_index: bool,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}

        if any(k in needed for k in ("x", "y", "z")):
            from common.app_context import get_app_context
            ctx = get_app_context()
            if ctx is not None:
                pos = ctx.motion.get_position()
                values["x"] = int(pos.x)
                values["y"] = int(pos.y)
                values["z"] = int(pos.z)
            else:
                values["x"] = 0
                values["y"] = 0
                values["z"] = 0

        if "i" in needed:
            if index is not None:
                values["i"] = int(index)
            else:
                from common.app_context import get_app_context
                ctx = get_app_context()
                if ctx is not None and hasattr(ctx, "current_sample_index"):
                    values["i"] = int(ctx.current_sample_index)
                else:
                    values["i"] = int(self._index)
                    if auto_increment_index:
                        self._index += 1

        return values

    def _render(
        self,
        template: str,
        values: dict[str, Any],
        full_widths: dict[str, int],
        strip_zeros: int,
    ) -> str:
        s = template.replace("{{", _TOKEN_OPEN).replace("}}", _TOKEN_CLOSE)
        recognized = {"x", "y", "z", "i", "d"}

        def _sub(m: re.Match) -> str:
            raw = m.group(1).strip()
            parts = raw.split(":", 1)
            key = parts[0].strip()

            if key not in recognized:
                return m.group(0)

            if key in ("x", "y", "z"):
                if key not in values:
                    return m.group(0)
                if self.pad_positions:
                    return _format_nm(values[key], full_widths[key], strip_zeros)
                divisor = 10 ** strip_zeros
                return str(values[key] // divisor)

            if key == "i":
                if key not in values:
                    return m.group(0)
                return str(values[key])

            if key == "d":
                fmt = parts[1] if len(parts) == 2 and parts[1] else self._default_date_format
                try:
                    return datetime.now().strftime(fmt)
                except Exception:
                    return m.group(0)

            return m.group(0)

        s = _FIELD_RE.sub(_sub, s)
        return s.replace(_TOKEN_OPEN, "{").replace(_TOKEN_CLOSE, "}")

    # ---------------- Public API ----------------

    def get_formatted_string(
        self,
        *,
        template: str | None = None,
        index: int | None = None,
        auto_increment_index: bool = False,
    ) -> str:
        """
        Build the formatted filename string.

        Args (all keyword-only, all optional):
            template:               Override the saved template for this call.
            index:                  Explicit index override.
            auto_increment_index:   Increment the internal index after use.
        """
        tpl = template if template is not None else self._template
        if not tpl:
            raise ValueError("No template provided or saved. Call set_template(...) or pass template=...")

        needed = self._needed_fields(tpl)
        full_widths = self._axis_full_widths(needed) if self.pad_positions else {"x": 0, "y": 0, "z": 0}
        strip_zeros = self._strip_zeros() if any(k in needed for k in ("x", "y", "z")) else 0

        values = self._collect_values(
            needed,
            index=index,
            auto_increment_index=auto_increment_index,
        )
        return self._render(tpl, values, full_widths, strip_zeros)

    def validate_template(self, template: str, *, strict: bool = False) -> dict[str, Any]:
        """Return a structured validation report for the given template."""
        issues: list[str] = []

        s = template.replace("{{", _TOKEN_OPEN).replace("}}", _TOKEN_CLOSE)
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    issues.append(f"Unmatched '}}' at position {i}.")
                    break
        if depth != 0:
            issues.append("Unbalanced braces detected.")

        recognized_keys = {"x", "y", "z", "i", "d"}
        seen_recognized: list[str] = []
        seen_unknown: list[str] = []
        date_tokens: list[dict[str, Any]] = []
        uses_default_date = False
        needs_positions = needs_index = needs_date = False

        for m in _FIELD_RE.finditer(template):
            raw = m.group(1).strip()
            parts = raw.split(":", 1)
            key = parts[0].strip()
            has_format = len(parts) == 2

            if key not in recognized_keys:
                if strict:
                    issues.append(f"Unknown placeholder {{{raw}}} at pos {m.start()}.")
                if key not in seen_unknown:
                    seen_unknown.append(key)
                continue

            if key not in seen_recognized:
                seen_recognized.append(key)

            if key in ("x", "y", "z"):
                needs_positions = True
            elif key == "i":
                needs_index = True
            elif key == "d":
                needs_date = True

            if key == "d":
                if has_format:
                    fmt = parts[1]
                    ok = True
                    try:
                        datetime.now().strftime(fmt)
                    except Exception as e:
                        ok = False
                        issues.append(f"Invalid date format in {{{raw}}}: {e}")
                    date_tokens.append({"raw": raw, "format": fmt, "valid": ok})
                else:
                    uses_default_date = True
                    date_tokens.append({"raw": raw, "format": self._default_date_format, "valid": True})
            elif has_format:
                issues.append(f"Formatting is only supported for {{d}}. Found on {{{raw}}}.")

        needed_set = self._needed_fields(template)
        pad_enabled = bool(self.pad_positions)
        full_widths = self._axis_full_widths(needed_set) if pad_enabled else {"x": 0, "y": 0, "z": 0}
        strip_zeros = self._strip_zeros()

        display_widths = {
            axis: max(0, full_widths[axis] - strip_zeros)
            for axis in ("x", "y", "z")
        }

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "placeholders": {"recognized": seen_recognized, "unknown": seen_unknown},
            "requires": {
                "positions": needs_positions,
                "index": needs_index,
                "date": needs_date,
            },
            "date": {"tokens": date_tokens, "uses_default_format": uses_default_date},
            "padding": {
                "enabled": pad_enabled,
                "full_widths_nm": full_widths,
                "strip_trailing_zeros": strip_zeros,
                "display_widths": display_widths,
            },
        }

    def is_template_valid(self, template: str, *, strict: bool = False) -> bool:
        return self.validate_template(template, strict=strict)["is_valid"]