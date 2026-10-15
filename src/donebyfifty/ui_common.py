"""Shared terminal UI helpers: console, prompts, parsing, formatting and banner."""

from __future__ import annotations

import math
from typing import (
    Any,
)

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .config import THEME_COLOR, THEME_COLOR_ACCENT, THEME_COLOR_BRIGHT, THEME_COLOR_ERROR

# =============================================================================
# WILSON SCORE INTERVAL (H8b — binomial uncertainty on p_success)
# =============================================================================


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Args:
        p: Observed proportion (0–1).
        n: Number of trials.
        z: Z-score for the desired confidence level (1.96 = 95%).

    Returns:
        (lower, upper) bounds of the interval, clamped to [0, 1].

    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# =============================================================================
# CONSOLE
# =============================================================================

console = Console(style=THEME_COLOR)

# =============================================================================
# ASCII BANNER
# =============================================================================

BANNER = r"""    ____                   ____        __________
   / __ \____  ____  ___  / __ )__  __/ ____/ __ \
  / / / / __ \/ __ \/ _ \/ __  / / / /___ \/ / / /
 / /_/ / /_/ / / / /  __/ /_/ / /_/ /___/ / /_/ /
/_____/\____/_/ /_/\___/_____/\__, /_____/\____/
                             /____/

            DoneBy50 Retirement Simulator"""


def print_banner() -> None:
    """Print the ASCII art banner."""
    banner_text = Text(BANNER, style=THEME_COLOR_BRIGHT)
    console.print(banner_text)
    console.print()


# =============================================================================
# HELPERS
# =============================================================================


def _parse_monetary(value: str) -> float | None:
    """Parse a number with optional '$', thousands commas, '%' and k/m/b shorthand.

    Examples: '300k' -> 300000, '1.5M' -> 1500000, '$500,000' -> 500000.0,
    '6.5%' -> 6.5 (the caller decides the unit of a percentage).

    Args:
        value: The input string to parse.

    Returns:
        A finite float if parsing succeeds, None otherwise (including 'inf' and 'nan').

    """
    cleaned = value.strip().lower().replace("$", "").replace(",", "").replace("%", "").strip()
    multiplier = 1.0
    if cleaned.endswith("k"):
        multiplier, cleaned = 1_000.0, cleaned[:-1]
    elif cleaned.endswith("m"):
        multiplier, cleaned = 1_000_000.0, cleaned[:-1]
    elif cleaned.endswith("b"):
        multiplier, cleaned = 1_000_000_000.0, cleaned[:-1]
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed * multiplier


def _range_message(lo: float | None, hi: float | None, *, whole: bool) -> str:
    """Return the rejection message for an unparsable or out-of-range number."""
    kind = "a whole number" if whole else "a number"
    if lo is not None and hi is not None:
        return f"Please enter {kind} between {lo:,.10g} and {hi:,.10g}"
    if lo is not None:
        return f"Please enter {kind} of at least {lo:,.10g}"
    if hi is not None:
        return f"Please enter {kind} of at most {hi:,.10g}"
    return f"Please enter {kind}"


def _ask_number(
    prompt_text: str,
    default: float | None,
    lo: float | None,
    hi: float | None,
    *,
    whole: bool,
) -> float:
    """Prompt until the user enters an in-range finite number; blank accepts the default."""
    default_str = None if default is None else f"{default:,.10g}"
    while True:
        raw = Prompt.ask(f"[{THEME_COLOR_BRIGHT}]{prompt_text}[/]", default=default_str)
        if raw is None or raw.strip() == "":
            if default is not None:
                return default
            continue
        parsed = _parse_monetary(raw)
        if parsed is None or (whole and not parsed.is_integer()):
            console.print(f"  [bold {THEME_COLOR_ERROR}]{_range_message(lo, hi, whole=whole)}[/]")
            continue
        if (lo is not None and parsed < lo) or (hi is not None and parsed > hi):
            console.print(f"  [bold {THEME_COLOR_ERROR}]{_range_message(lo, hi, whole=whole)}[/]")
            continue
        return parsed


def _prompt_int(
    prompt_text: str,
    default: int | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> int:
    """Prompt for a whole number with optional default and range validation."""
    default_f = None if default is None else float(default)
    return int(_ask_number(prompt_text, default_f, lo, hi, whole=True))


def _prompt_float(
    prompt_text: str,
    default: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Prompt for a number with optional default and range validation."""
    return _ask_number(prompt_text, default, lo, hi, whole=False)


def _prompt_pct(
    prompt_text: str,
    default: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Prompt for a percentage and return it as a decimal fraction (6.5 -> 0.065).

    ``default``, ``lo`` and ``hi`` are in percent units. A bare entry below 1
    (e.g. "0.065") is confirmed as "Did you mean 6.5%?" before it is read as a
    percentage, so a fraction is never silently taken as 0.065%.
    """
    default_str = None if default is None else f"{default:,.10g}%"
    while True:
        raw = Prompt.ask(f"[{THEME_COLOR_BRIGHT}]{prompt_text}[/]", default=default_str)
        if raw is None or raw.strip() == "":
            if default is not None:
                return default / 100.0
            continue
        parsed = _parse_monetary(raw)
        if parsed is None:
            console.print(f"  [bold {THEME_COLOR_ERROR}]{_range_message(lo, hi, whole=False)}[/]")
            continue
        pct = parsed
        if "%" not in raw and 0.0 < abs(parsed) < 1.0:
            if not _prompt_yn(f"Did you mean {parsed * 100:g}%?", default=True):
                continue
            pct = parsed * 100.0
        if (lo is not None and pct < lo) or (hi is not None and pct > hi):
            console.print(f"  [bold {THEME_COLOR_ERROR}]{_range_message(lo, hi, whole=False)}[/]")
            continue
        return pct / 100.0


def _prompt_yn(prompt_text: str, default: bool = True) -> bool:
    """Prompt for yes/no, re-prompting until a valid answer or the default is given."""
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = Prompt.ask(
            f"[{THEME_COLOR_BRIGHT}]{prompt_text}[/] [{default_str}]",
            default="y" if default else "n",
        )
        answer = raw.strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        console.print(f"  [bold {THEME_COLOR_ERROR}]Please enter y or n[/]")


# =============================================================================
# PROFILE LISTING
# =============================================================================


def display_profile_list(profiles: list[dict[str, Any]]) -> None:
    """Display a list of saved profiles."""
    if not profiles:
        console.print("[yellow]No saved profiles.[/]")
        return

    table = Table(
        title="Saved Profiles",
        border_style=THEME_COLOR_ACCENT,
        box=None,
    )
    table.add_column("#", style=THEME_COLOR_BRIGHT)
    table.add_column("Name", style=THEME_COLOR)
    table.add_column("Earners", justify="right")
    table.add_column("Children", justify="right")
    table.add_column("Last Success", justify="right")

    for i, p in enumerate(profiles, 1):
        success_str = (
            f"{p['p_success'] * 100:.1f}%"
            if p.get("p_success") is not None
            else ("[red]corrupt[/]" if p.get("corrupt") else "—")
        )
        table.add_row(
            str(i),
            escape(p["name"]),
            str(p["num_earners"]),
            str(p["num_children"]),
            success_str,
        )

    console.print(table)
