"""Results display: summary, results menu and drill-down views."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .analysis import RetirementSearchResult
from .config import (
    BK_KAPPA,
    BK_RHO,
    BK_SIGMA_MODERATE,
    BK_THETA,
    DEFAULT_SIM_START_AGE,
    THEME_COLOR,
    THEME_COLOR_ACCENT,
    THEME_COLOR_BRIGHT,
    THEME_COLOR_ERROR,
    THEME_COLOR_WARN,
)
from .models import Household, MortgageAccount, ResultsSession, SimulationResults
from .ui_common import _prompt_int, _prompt_yn, _wilson_interval, console
from .ui_wizard import _display_contribution_warnings

_SUCCESS_DEFINITION = (
    "A trial fails if any year has unmet spending, or the net bridge balance is zero or below."
)
_GLOSSARY = (
    "P5 / P95: 5th / 95th percentile (1 in 20 trials falls below / above). "
    "SE: standard error. pp: percentage points."
)
_SE_LEGEND = (
    "SE %: standard error as a share of the estimate. "
    "Green < 5% (tight), yellow 5-20% (moderate), red >= 20% (loose)."
)


# =============================================================================
# FORMATTING HELPERS
# =============================================================================


def _fmt_money(x: float, *, compact: bool = False) -> str:
    """Format a dollar amount with the sign in front, e.g. ``-$1,234``.

    ``compact`` switches to millions (``$1.32M``) so one column can share a scale.
    """
    body = f"${abs(x) / 1e6:,.2f}M" if compact else f"${abs(x):,.0f}"
    if body in ("$0", "$0.00M"):
        return body
    return f"-{body}" if x < 0 else body


def _money_formatter(values: Sequence[float]) -> Callable[[float], str]:
    """Return a formatter for one table column.

    The column uses millions when any value reaches a million, so every row in
    that column is on the same scale.
    """
    compact = any(abs(v) >= 1e6 for v in values)
    return lambda x: _fmt_money(x, compact=compact)


def _fmt_share(x: float) -> str:
    """Format a proportion as a percentage. Tiny or near-total values are not rounded to 0% or 100%."""
    pct = x * 100.0
    if x > 0.0 and round(pct, 1) <= 0.0:
        return "<0.1%"
    if x < 1.0 and round(pct, 1) >= 100.0:
        return "≥99.9%"
    return f"{pct:.1f}%"


def _fmt_rate(x: float) -> str:
    """Format an annual interest rate (decimal) as a percentage with two decimals."""
    return f"{x * 100:.2f}%"


def _cell(series: Sequence[float], i: int, fmt: Callable[[float], str]) -> str:
    """Format ``series[i]``, or a dash when the series is shorter."""
    return fmt(series[i]) if i < len(series) else "—"


def _table(title: str | None = None) -> Table:
    """Build a borderless table that fits an 80-column terminal."""
    return Table(
        title=title,
        border_style=THEME_COLOR_ACCENT,
        box=None,
        pad_edge=False,
        collapse_padding=True,
    )


def _se_pct_cell(est: float, se: float) -> str:
    """Colour-graded SE as a share of the estimate."""
    if est == 0.0:
        return "[dim]n/a[/]"
    pct = abs(se / est) * 100.0
    if pct < 5.0:
        colour = THEME_COLOR
    elif pct < 20.0:
        colour = THEME_COLOR_WARN
    else:
        colour = THEME_COLOR_ERROR
    return f"[{colour}]{pct:.1f}%[/]"


def _print_total_quantiles(heading: str, rows: Sequence[tuple[str, float, float, float]]) -> None:
    """Print per-trial total quantiles, one line per series."""
    console.print(f"  [bold]{heading}[/]")
    for label, p5, p50, p95 in rows:
        console.print(
            f"    {label:<18} P5 {_fmt_money(p5)}  |  Median {_fmt_money(p50)}"
            f"  |  P95 {_fmt_money(p95)}"
        )


def _success_headline(p_success: float, trials: int, success_threshold: float) -> str:
    """One-line success summary with failure count, Wilson CI and threshold status."""
    threshold_pct = success_threshold * 100.0
    if p_success >= success_threshold:
        status = f"[{THEME_COLOR_BRIGHT}](threshold {threshold_pct:.0f}%: reached)[/]"
    else:
        status = f"[{THEME_COLOR_ERROR}](threshold {threshold_pct:.0f}%: not reached)[/]"
    ci_note = ""
    failure_note = ""
    if trials > 0:
        failures = round(trials * (1.0 - p_success))
        failure_note = f"  ({failures:,} of {trials:,} paths failed)"
        lo, hi = _wilson_interval(p_success, trials)
        ci_note = f"  [95% CI: {lo * 100:.1f}\u2013{hi * 100:.1f}%]"
    return (
        f"  Success probability: [bold]{_fmt_share(p_success)}[/]{ci_note}  {status}{failure_note}"
    )


def _bk_disclosure(mortgage: MortgageAccount) -> str:
    """Describe the Black-Karasinski rate model used for one stochastic mortgage.

    Stationary AR(1) on ln(r) is centred on ln(theta) - sigma^2/(4 kappa) with
    standard deviation sigma / sqrt(2 kappa), so the 95% band is exp(mu +/- 1.96 sd).
    """
    theta = mortgage.interest_rate_theta if mortgage.interest_rate_theta is not None else BK_THETA
    kappa = mortgage.interest_rate_kappa if mortgage.interest_rate_kappa is not None else BK_KAPPA
    sigma = (
        mortgage.interest_rate_vol if mortgage.interest_rate_vol is not None else BK_SIGMA_MODERATE
    )
    rho = mortgage.interest_rate_corr if mortgage.interest_rate_corr is not None else BK_RHO
    log_mean = math.log(theta) - sigma**2 / (4.0 * kappa)
    log_sd = sigma / math.sqrt(2.0 * kappa)
    lo = math.exp(log_mean - 1.96 * log_sd) * 100.0
    hi = math.exp(log_mean + 1.96 * log_sd) * 100.0
    return (
        f"{escape(mortgage.label)}: rates follow a Black\u2013Karasinski mean-reverting "
        f"log-normal process (discrete AR(1), kappa={kappa:.2f}/yr). The long-run arithmetic "
        f"mean is {theta * 100:.1f}% and the log-rate volatility is sigma={sigma:.2f}/sqrt(yr). "
        f"The stationary 95% band is between {lo:.1f}% and {hi:.1f}%. "
        f"Rate shocks are correlated with equity returns (rho={rho:.2f})."
    )


def _offset_median_series(
    results: SimulationResults, labels: Sequence[str], n: int
) -> list[float] | None:
    """Per-year sum of the median balances of the linked offset accounts.

    This is a sum of medians, not the median of the total, because the
    percentile paths are not joint across accounts.
    """
    series = [
        results.offset_by_age[label]["p50"] for label in labels if label in results.offset_by_age
    ]
    if not series:
        return None
    return [sum(s[i] for s in series if i < len(s)) for i in range(n)]


# =============================================================================
# SUMMARY SCREEN
# =============================================================================


def display_results(
    results: SimulationResults,
    household: Household | None = None,
    start_age: int = DEFAULT_SIM_START_AGE,
    success_threshold: float = 0.95,
) -> None:
    """Print the main simulation summary (fits an 80-column terminal)."""
    console.print()
    console.print(
        Panel(
            f"[bold {THEME_COLOR_BRIGHT}]Simulation Complete \u2014 {results.trials:,} trials[/]",
            border_style=THEME_COLOR_ACCENT,
        )
    )
    console.print()
    console.print(_success_headline(results.p_success, results.trials, success_threshold))
    console.print(f"  [dim]{_SUCCESS_DEFINITION}[/]")
    console.print(f"  [dim]{_GLOSSARY}[/]")

    # Plain-English risk tier
    if results.p_success == 0.0:
        console.print(
            f"  \u26a0\ufe0f  [bold {THEME_COLOR_ERROR}]Every simulated path failed. "
            "Your bridge assets are not enough to fund the plan.[/]"
        )
        if results.bridge_median < 0:
            console.print(f"  [dim]Median shortfall: {_fmt_money(abs(results.bridge_median))}[/]")
    elif results.p_success < 0.30:
        console.print(
            f"  \u26a0\ufe0f  [bold {THEME_COLOR_ERROR}]Severe risk: success in only "
            f"{_fmt_share(results.p_success)} of trials. Your bridge is far too tight. "
            "Consider more non-super savings, lower spending, or a shorter bridge period.[/]"
        )
    elif results.p_success < 0.50:
        console.print(
            f"  \u26a0\ufe0f  [bold {THEME_COLOR_ERROR}]High risk: success in "
            f"{_fmt_share(results.p_success)} of trials. More than half of possible futures "
            "run out early. Increase bridge assets or reduce expenses.[/]"
        )
    elif results.p_success < 0.70:
        console.print(
            f"  \u26a0\ufe0f  [bold {THEME_COLOR_WARN}]Moderate risk: success in "
            f"{_fmt_share(results.p_success)} of trials. Consider boosting non-super savings "
            "or trimming expenses to improve odds.[/]"
        )
    elif results.p_success < success_threshold:
        console.print(
            f"  \u26a0\ufe0f  [{THEME_COLOR_WARN}]Success probability {_fmt_share(results.p_success)} "
            f"is below the {success_threshold * 100:.0f}% threshold. A small buffer would improve "
            "confidence.[/]"
        )

    # Bridge balances
    bridge_rows = [
        ("Mean", results.bridge_mean),
        ("Median", results.bridge_median),
        ("P5 (1 in 20 worse)", results.bridge_p5),
        ("P95 (1 in 20 better)", results.bridge_p95),
    ]
    bridge_fmt = _money_formatter([v for _, v in bridge_rows])
    console.print()
    table = _table("Bridge Assets at Horizon (today's dollars)")
    table.add_column("Metric", style=THEME_COLOR_BRIGHT)
    table.add_column("Value", justify="right", style=THEME_COLOR, no_wrap=True)
    for label, value in bridge_rows:
        table.add_row(label, bridge_fmt(value))
    console.print(table)

    # Standard errors (only where the run computed them)
    se_rows: list[tuple[str, float, float, str]] = []
    for label, est, se, method in (
        ("Mean", results.bridge_mean, results.bridge_mean_se, "analytic"),
        ("Median", results.bridge_median, results.bridge_median_se, "bootstrap"),
        ("P5", results.bridge_p5, results.bridge_p5_se, "bootstrap"),
        ("P95", results.bridge_p95, results.bridge_p95_se, "bootstrap"),
    ):
        if se is not None:
            se_rows.append((label, est, se, method))
    if se_rows:
        se_fmt = _money_formatter([x for _, est, se, _m in se_rows for x in (est, se)])
        console.print()
        se_table = _table("Uncertainty (today's dollars)")
        se_table.add_column("Metric", style=THEME_COLOR_BRIGHT)
        se_table.add_column("Estimate", justify="right", style=THEME_COLOR, no_wrap=True)
        se_table.add_column("SE", justify="right", style=THEME_COLOR, no_wrap=True)
        se_table.add_column("SE %", justify="right", no_wrap=True)
        se_table.add_column("Method", style="dim")
        for label, est, se, method in se_rows:
            se_table.add_row(label, se_fmt(est), se_fmt(se), _se_pct_cell(est, se), method)
        console.print(se_table)
        console.print(f"  [dim]{_SE_LEGEND}[/]")

    # Worst outcome (only meaningful with enough trials)
    if results.trials >= 1000:
        console.print()
        console.print(
            Panel(
                f"[bold {THEME_COLOR_ERROR}]Worst outcome (1 in {results.trials:,})[/]\n"
                f"  Lowest bridge balance at age {results.horizon_age}: "
                f"[bold]{_fmt_money(results.bridge_floor)}[/]\n"
                f"  [dim]The lowest outcome across all trials. Shows how bad the "
                f"tail can get.[/]",
                border_style=THEME_COLOR_ERROR,
            )
        )

    # Super
    console.print()
    console.print(
        f"  Median super balance at end of scenario (today's dollars): "
        f"[bold]{_fmt_money(results.super_median)}[/]"
    )
    if results.per_earner_super_p50:
        super_fmt = _money_formatter(list(results.per_earner_super_p50.values()))
        super_table = _table("Median Super per Earner (today's dollars)")
        super_table.add_column("Earner", style=THEME_COLOR_BRIGHT)
        super_table.add_column("Median super", justify="right", style=THEME_COLOR, no_wrap=True)
        for earner, value in results.per_earner_super_p50.items():
            super_table.add_row(escape(earner), super_fmt(value))
        console.print(super_table)

    # Mortgage balances (nominal)
    if results.remaining_mortgage_p50:
        mortgage_fmt = _money_formatter(list(results.remaining_mortgage_p50.values()))
        console.print()
        mort_table = _table("Remaining Mortgage (median, nominal dollars)")
        mort_table.add_column("Mortgage", style=THEME_COLOR_BRIGHT)
        mort_table.add_column(
            "Balance at horizon", justify="right", style=THEME_COLOR, no_wrap=True
        )
        for label, value in results.remaining_mortgage_p50.items():
            mort_table.add_row(escape(label), mortgage_fmt(value))
        console.print(mort_table)

    # Mortgage term clearance
    if results.per_mortgage_term_cleared_pct:
        tc_rate = results.mortgage_term_clearance_rate
        tc_color = (
            THEME_COLOR_BRIGHT
            if tc_rate >= 0.9
            else THEME_COLOR_ERROR if tc_rate < 0.5 else THEME_COLOR_WARN
        )
        console.print()
        console.print(
            f"  Mortgage term clearance: [bold {tc_color}]{tc_rate * 100:.1f}%[/] "
            "of paths cleared by term end"
        )
        if tc_rate < 1.0:
            for label, pct in results.per_mortgage_term_cleared_pct.items():
                if pct < 1.0:
                    console.print(
                        f"    [dim]{escape(label)}: cleared in {pct * 100:.1f}% of paths[/]"
                    )
            console.print(
                "  [dim]Mortgages still outstanding at term end are not cleared by the "
                "modelled horizon.[/]"
            )

    # Mortgages that cannot be checked
    if results.per_mortgage_not_checkable:
        console.print()
        for label, count in results.per_mortgage_not_checkable.items():
            console.print(
                f"  [dim]\u26a0\ufe0f  {escape(label)}: term end is beyond the simulation "
                f"horizon (age {results.horizon_age}). Clearance cannot be checked in "
                f"{count:,} of {results.trials:,} trials.[/]"
            )
    elif results.remaining_mortgage_p50 and not results.per_mortgage_term_cleared_pct:
        console.print()
        console.print("  [dim]No term-end check configured for the mortgages in this run.[/]")

    # Contribution warnings
    if household is not None:
        _display_contribution_warnings(household, start_age)

    # Warnings raised by the simulation itself
    for warning in results.warnings:
        console.print(f"  [{THEME_COLOR_WARN}]{escape(warning)}[/]")

    # Disclosures
    console.print()
    disclosures = [
        "Bridge assets, super and drawdown figures are today's dollars (deflated to "
        "start-of-simulation purchasing power). Mortgage and offset balances are nominal "
        "(then-year) dollars.",
        f"This simulation models only the bridge period (up to age {results.horizon_age}). "
        "It does not model post-bridge retirement drawdown, Age Pension, or minimum "
        "pension rules.",
    ]
    if (
        household is not None
        and results.remaining_mortgage_p50
        and any(v > 0 for v in results.remaining_mortgage_p50.values())
    ):
        for mortgage in household.mortgages:
            if mortgage.interest_rate_stochastic:
                disclosures.append(_bk_disclosure(mortgage))
        if not all(m.interest_rate_stochastic for m in household.mortgages):
            disclosures.append(
                "Mortgages with fixed rates hold their rate constant for the full simulation. "
                "Rate volatility is not modelled for them, which may understate repayment "
                "stress in rising-rate environments."
            )
    for i, note in enumerate(disclosures):
        if i > 0:
            console.print()
        console.print(f"  [dim]Note: {note}[/]")
    console.print()


# =============================================================================
# RESULTS MENU
# =============================================================================


def show_results_menu(session: ResultsSession) -> None:
    """Display the results menu and dispatch to drill-down views."""
    while True:
        console.print()
        console.print(
            Panel(
                f"Simulation complete \u2014 {session.results.trials:,} trials\n"
                f"Success: {_fmt_share(session.results.p_success)}"
                + (f"\nSeed: {session.inputs.seed}" if session.inputs.seed is not None else ""),
                border_style=THEME_COLOR_ACCENT,
            )
        )
        options = [
            ("1", "Age-by-age bridge trajectory"),
            ("2", "Near misses & failure timing"),
            ("3", "Sequencing risk (opt-in, ~2 min)"),
            ("4", "Drawdown source composition"),
            ("5", "CGT breakdown"),
            ("6", "Scenario comparison"),
            ("7", "Earliest feasible retirement age"),
            ("8", "Mortgage amortisation"),
            ("0", "Back to main menu"),
        ]
        for key, label in options:
            console.print(f"  [bold {THEME_COLOR_BRIGHT}]{key}[/]  {label}")
        console.print()

        choice = Prompt.ask("Choose", choices=[k for k, _ in options], default="0")
        if choice == "0":
            return
        if choice == "1":
            _view_trajectory(session)
        elif choice == "2":
            _view_near_miss(session)
        elif choice == "3":
            _view_sequencing_risk(session)
        elif choice == "4":
            _view_drawdown_composition(session)
        elif choice == "5":
            _view_cgt_breakdown(session)
        elif choice == "6":
            _view_scenario_comparison(session)
        elif choice == "7":
            _view_retirement_search(session)
        elif choice == "8":
            _view_mortgage_amortisation(session)


# =============================================================================
# VIEWS
# =============================================================================


def _view_trajectory(session: ResultsSession) -> None:
    """Age-by-age bridge percentile table."""
    r = session.results
    ages = r.bridge_by_age_ages
    if not ages:
        console.print("  [dim]No trajectory data available.[/]")
        return

    series: list[tuple[str, Sequence[float], str]] = [
        ("P5", r.bridge_by_age_p5, THEME_COLOR),
        ("P10", r.bridge_by_age_p10, THEME_COLOR),
        ("P25", r.bridge_by_age_p25, THEME_COLOR),
        ("Median", r.bridge_by_age_p50, THEME_COLOR_BRIGHT),
        ("P75", r.bridge_by_age_p75, THEME_COLOR),
        ("P90", r.bridge_by_age_p90, THEME_COLOR),
        ("P95", r.bridge_by_age_p95, THEME_COLOR),
    ]
    fmts = [_money_formatter(values) for _, values, _ in series]

    console.print()
    table = _table("Bridge trajectory (today's dollars)")
    table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
    for name, _, style in series:
        table.add_column(name, justify="right", style=style, no_wrap=True)
    for i, age in enumerate(ages):
        table.add_row(
            str(age),
            *(_cell(values, i, fmt) for fmt, (_, values, _) in zip(fmts, series, strict=True)),
        )
    console.print(table)
    console.print()
    console.print("  [dim]Copy this table into Excel or Google Sheets for further analysis.[/]")


def _view_near_miss(session: ResultsSession) -> None:
    """Failure rate, near-miss bands and failure-age timing."""
    r = session.results
    failures = round(r.trials * (1.0 - r.p_success))
    console.print()
    console.print("  [bold]Near misses & failure timing[/]")
    console.print(
        f"  Failed trials: {failures:,} of {r.trials:,} ({_fmt_share(1.0 - r.p_success)})"
    )
    if r.near_miss_threshold > 0:
        console.print(
            f"  Near-miss band: bridge floor between $0 and {_fmt_money(r.near_miss_threshold)} "
            "(about one year of your retirement spending target, today's dollars)."
        )
        console.print(
            f"  Near-miss rate (failed, or floor within band): {_fmt_share(r.near_miss_rate)}"
        )
        console.print(
            f"  Floor within band, whether failed or not: {_fmt_share(r.near_miss_only_rate)}"
        )
    else:
        console.print("  [dim]No near-miss band is set for this run.[/]")

    if r.failure_age_distribution:
        total_failures = sum(r.failure_age_distribution.values())
        console.print()
        console.print(f"  [bold]Failure age distribution[/] ({total_failures:,} failing trials)")
        table = _table()
        table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
        table.add_column("Failures", justify="right", style=THEME_COLOR, no_wrap=True)
        table.add_column("% of failures", justify="right", style=THEME_COLOR, no_wrap=True)
        for age in sorted(r.failure_age_distribution):
            count = r.failure_age_distribution[age]
            table.add_row(str(age), f"{count:,}", f"{count / total_failures * 100:.1f}%")
        console.print(table)
    else:
        console.print("  [dim]No failures occurred in this simulation run.[/]")


def _view_sequencing_risk(session: ResultsSession) -> None:
    """Sequencing-risk analysis (worst-first vs best-first return order)."""
    from .analysis import run_sequencing_analysis

    seq = session.sequencing
    if seq is None:
        if not _prompt_yn(
            "Run sequencing risk analysis? This will run 2 additional "
            "full simulations and may take 2-3 minutes.",
            default=False,
        ):
            return
        with console.status("Running sequencing risk analysis (2 passes)..."):
            seq = run_sequencing_analysis(inputs=session.inputs, seed=session.inputs.seed)
        session.sequencing = seq

    console.print()
    console.print("  [bold]Sequencing risk analysis[/]")
    console.print()
    console.print(
        f"  Original (random order):   {_fmt_share(seq.original_p_success)} success,  "
        f"P5 = {_fmt_money(seq.original_p5)}"
    )
    console.print(
        f"  Worst returns first:       {_fmt_share(seq.worst_first_p_success)} success,  "
        f"P5 = {_fmt_money(seq.worst_first_p5)}"
    )
    console.print(
        f"  Best returns first:        {_fmt_share(seq.best_first_p_success)} success,  "
        f"P5 = {_fmt_money(seq.best_first_p5)}"
    )
    console.print()
    console.print(
        f"  Sequencing risk spread:    [bold]{seq.spread * 100:.1f} pp[/] "
        "(best-first minus worst-first success)"
    )
    console.print()
    console.print("  [bold]What this means[/]")
    console.print(
        "  The simulated return sequence is reordered so the worst years come first in "
        "your bridge period."
    )
    console.print(
        "  If the spread is large, the timing of bad markets matters more than the average "
        "return."
    )
    console.print(
        "  Consider a cash buffer or lower equity exposure early in the bridge period "
        "if the spread is large."
    )


def _view_drawdown_composition(session: ResultsSession) -> None:
    """Median per-year drawdown sources, CGT and investment-income tax."""
    r = session.results
    ages = r.bridge_by_age_ages

    if not r.offset_drawn_p50 or not ages:
        console.print("  [dim]No drawdown composition data available.[/]")
        return

    n_rows = min(
        len(ages), len(r.offset_drawn_p50), len(r.non_offset_drawn_p50), len(r.cgt_paid_p50)
    )
    series: list[tuple[str, Sequence[float]]] = [
        ("Offset drawn", r.offset_drawn_p50),
        ("Non-offset drawn", r.non_offset_drawn_p50),
        ("CGT paid", r.cgt_paid_p50),
    ]
    # Skip years with no drawdown activity, but always keep the final year.
    keep = [
        i for i in range(n_rows) if i == n_rows - 1 or any(vals[i] != 0.0 for _, vals in series)
    ]
    fmts = [_money_formatter([vals[i] for i in keep]) for _, vals in series]

    console.print()
    console.print("  [bold]Drawdown source composition (median per year, today's dollars)[/]")
    table = _table()
    table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
    for name, _ in series:
        table.add_column(name, justify="right", style=THEME_COLOR, no_wrap=True)
    for i in keep:
        table.add_row(
            str(ages[i]),
            *(fmt(vals[i]) for fmt, (_, vals) in zip(fmts, series, strict=True)),
        )
    console.print(table)

    console.print()
    if any(
        v != 0.0
        for v in (r.offset_drawn_total_p50, r.non_offset_drawn_total_p50, r.cgt_paid_total_p50)
    ):
        _print_total_quantiles(
            "Total across bridge (per-trial quantiles, today's dollars):",
            [
                (
                    "Offset drawn",
                    r.offset_drawn_total_p5,
                    r.offset_drawn_total_p50,
                    r.offset_drawn_total_p95,
                ),
                (
                    "Non-offset drawn",
                    r.non_offset_drawn_total_p5,
                    r.non_offset_drawn_total_p50,
                    r.non_offset_drawn_total_p95,
                ),
                (
                    "CGT paid",
                    r.cgt_paid_total_p5,
                    r.cgt_paid_total_p50,
                    r.cgt_paid_total_p95,
                ),
            ],
        )
    else:
        console.print(
            "  [bold]Total (sum of per-year medians, not a true quantile):[/] "
            f"Offset: {_fmt_money(sum(r.offset_drawn_p50))}  |  "
            f"Non-offset: {_fmt_money(sum(r.non_offset_drawn_p50))}  |  "
            f"CGT paid: {_fmt_money(sum(r.cgt_paid_p50))}"
        )

    _print_income_tax(r, ages)

    console.print()
    console.print(
        "  [dim]CGT is charged only on real (CPI-indexed) gains. Drawdown is taken from the "
        "offset first, then non-offset accounts.[/]"
    )


def _print_income_tax(r: SimulationResults, ages: Sequence[int]) -> None:
    """Investment-income tax per year and in total, shown only when the run captured it."""
    if not r.income_tax_p50:
        return
    n_rows = min(len(ages), len(r.income_tax_p50))
    keep = [i for i in range(n_rows) if r.income_tax_p50[i] != 0.0]
    console.print()
    if keep:
        tax_fmt = _money_formatter([r.income_tax_p50[i] for i in keep])
        console.print("  [bold]Investment-income tax (median per year, today's dollars)[/]")
        table = _table()
        table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
        table.add_column("Median tax", justify="right", style=THEME_COLOR, no_wrap=True)
        for i in keep:
            table.add_row(str(ages[i]), tax_fmt(r.income_tax_p50[i]))
        console.print(table)
    totals = (r.income_tax_total_p5, r.income_tax_total_p50, r.income_tax_total_p95)
    if any(v != 0.0 for v in totals):
        _print_total_quantiles(
            "Total investment-income tax (per-trial quantiles, today's dollars):",
            [("Income tax", *totals)],
        )


def _view_cgt_breakdown(session: ResultsSession) -> None:
    """CGT paid per year with and without the 30% tax-rate floor."""
    r = session.results
    ages = r.bridge_by_age_ages

    if not r.cgt_paid_p50 or not ages:
        console.print("  [dim]No CGT data available (no taxable disposals in this run).[/]")
        return

    n_rows = min(len(ages), len(r.cgt_paid_p50), len(r.cgt_without_floor_p50))
    rows = [
        i
        for i in range(n_rows)
        if not (r.cgt_paid_p50[i] == 0.0 and r.cgt_without_floor_p50[i] == 0.0)
    ]
    with_floor = [r.cgt_paid_p50[i] for i in rows]
    without_floor = [r.cgt_without_floor_p50[i] for i in rows]
    extra = [a - b for a, b in zip(with_floor, without_floor, strict=True)]
    fmt_with = _money_formatter(with_floor)
    fmt_without = _money_formatter(without_floor)
    fmt_extra = _money_formatter(extra)

    console.print()
    table = _table("CGT breakdown (median per year, today's dollars)")
    table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
    table.add_column("With 30% floor", justify="right", style=THEME_COLOR, no_wrap=True)
    table.add_column("Without floor", justify="right", style=THEME_COLOR, no_wrap=True)
    table.add_column("Floor cost", justify="right", style=THEME_COLOR_WARN, no_wrap=True)
    for i, a, b, d in zip(rows, with_floor, without_floor, extra, strict=True):
        table.add_row(str(ages[i]), fmt_with(a), fmt_without(b), fmt_extra(d))
    console.print(table)

    console.print()
    if any(v != 0.0 for v in (r.cgt_paid_total_p50, r.cgt_without_floor_total_p50)):
        _print_total_quantiles(
            "Total CGT across bridge (per-trial quantiles, today's dollars):",
            [
                (
                    "With 30% floor",
                    r.cgt_paid_total_p5,
                    r.cgt_paid_total_p50,
                    r.cgt_paid_total_p95,
                ),
                (
                    "Without floor",
                    r.cgt_without_floor_total_p5,
                    r.cgt_without_floor_total_p50,
                    r.cgt_without_floor_total_p95,
                ),
                (
                    "Extra from floor",
                    r.cgt_floor_extra_total_p5,
                    r.cgt_floor_extra_total_p50,
                    r.cgt_floor_extra_total_p95,
                ),
            ],
        )
        if r.cgt_floor_extra_total_p50 > 0 and r.cgt_without_floor_total_p50 > 0:
            pct_increase = r.cgt_floor_extra_total_p50 / r.cgt_without_floor_total_p50 * 100
            console.print(
                f"  [dim](median extra CGT = {pct_increase:.1f}% increase over "
                "marginal-rate-only)[/]"
            )
    else:
        total_with_floor = sum(r.cgt_paid_p50)
        total_without_floor = sum(r.cgt_without_floor_p50)
        floor_cost = total_with_floor - total_without_floor
        if floor_cost > 0:
            console.print("  [bold]Total CGT (sum of per-year medians, not a true quantile):[/]")
            console.print(f"    With 30% floor:   {_fmt_money(total_with_floor)}")
            console.print(f"    Without floor:    {_fmt_money(total_without_floor)}")
            console.print(f"    Floor cost:       {_fmt_money(floor_cost)}")
        else:
            console.print("  [dim]The 30% floor has no effect in this run.[/]")


def _fmt_change(delta_pp: float, half_width_pp: float) -> str:
    """Scenario change as ``+x.x pp ± y.y``, flagged when inside the noise band."""
    text = f"{delta_pp:+.1f} pp \u00b1 {half_width_pp:.1f}"
    if abs(delta_pp) < half_width_pp:
        return f"{text} [dim](within noise)[/]"
    return text


def _view_scenario_comparison(session: ResultsSession) -> None:
    """Display scenario comparison table (opt-in, expensive).

    Scenarios share the base case's random draws, so each change is a paired difference.
    """
    from .analysis import run_scenario_comparison

    inputs = session.inputs
    earners = session.household.earners
    if session.scenarios is not None:
        scens = session.scenarios
    else:
        # Count applicable scenarios dynamically
        n_scens = 1  # expenses 10% higher
        n_scens += 1  # expenses 10% lower
        n_scens += len(earners)  # stops working
        if any(e.pt_days_per_week > 0 for e in earners):
            n_scens += 1  # no PT income

        est_min = max(1, n_scens // 2)
        if not _prompt_yn(
            f"Run {n_scens} scenario comparison{'s' if n_scens > 1 else ''}?"
            f" This will run {n_scens} additional simulations"
            f" and may take ~{est_min}-{n_scens} minutes.",
            default=False,
        ):
            return

        with console.status(f"Running {n_scens} scenario comparisons..."):
            scens = run_scenario_comparison(inputs=inputs, seed=inputs.seed)
        session.scenarios = scens

    base_p = session.results.p_success
    base_p5 = session.results.bridge_p5
    all_p5 = [base_p5, *(r.bridge_p5 for r in scens.values())]
    show_p5 = any(round(v) != 0 for v in all_p5)
    p5_fmt = _money_formatter(all_p5)

    console.print()
    table = _table(f"Scenario comparison (base {session.results.trials:,} trials)")
    table.add_column("Scenario", style=THEME_COLOR_BRIGHT)
    table.add_column("Success", justify="right", style=THEME_COLOR, no_wrap=True)
    table.add_column("Change (95% CI)", justify="right", no_wrap=True)
    if show_p5:
        table.add_column("P5 bridge", justify="right", style=THEME_COLOR, no_wrap=True)
    base_row = ["Current plan", _fmt_share(base_p), "\u2014"]
    if show_p5:
        base_row.append(p5_fmt(base_p5))
    table.add_row(*base_row)
    for label, r in scens.items():
        row = [
            escape(label),
            _fmt_share(r.p_success),
            _fmt_change(r.delta_p * 100.0, 1.96 * r.delta_se * 100.0),
        ]
        if show_p5:
            row.append(p5_fmt(r.bridge_p5))
        table.add_row(*row)
    console.print(table)
    console.print()
    console.print(
        "  [dim]Scenarios share the base case's simulated paths, so each change is a paired "
        "difference. \u00b1 is the 95% CI half-width. pp = percentage points.[/]"
    )


def _view_retirement_search(session: ResultsSession) -> None:
    """Display earliest feasible retirement age (opt-in, expensive).

    Single-earner: runs both_together automatically.
    Multi-earner: prompts for mode choice (both_together vs per_earner).
    """
    from .analysis import run_retirement_search

    inputs = session.inputs
    threshold = inputs.success_threshold
    earners = session.household.earners

    # Single earner: no mode choice needed
    if len(earners) == 1:
        rs = session.retirement_search_all
        if rs is None or rs.mode != "both_together":
            if not _prompt_yn(
                "Search for earliest feasible retirement age? This will run "
                "several full simulations and may take 2-3 minutes.",
                default=False,
            ):
                return
            with console.status("Searching for earliest feasible retirement age..."):
                rs = run_retirement_search(
                    inputs=inputs, seed=inputs.seed, success_threshold=threshold
                )
            session.retirement_search_all = rs
        _display_retirement_search_result(rs, main_trials=session.results.trials)
        return

    # Multi-earner: prompt for mode
    mode_names = {
        "both_together": "Both/all retire at same earliest age",
        "per_earner": "Pick one earner to optimise (others fixed)",
    }
    while True:
        console.print()
        console.print(f"  [{THEME_COLOR_BRIGHT}]Earliest-retirement search mode:[/]")
        console.print(f"    [{THEME_COLOR}]1.[/] {mode_names['both_together']}")
        console.print(f"    [{THEME_COLOR}]2.[/] {mode_names['per_earner']}")
        mode_choice = _prompt_int("  Choice", default=1, lo=1, hi=2)

        # Both together
        if mode_choice == 1:
            rs = session.retirement_search_all
            if rs is None or rs.mode != "both_together":
                if not _prompt_yn(
                    "Search for earliest age BOTH earners can retire? "
                    "This will run several full simulations and may take 2-3 minutes.",
                    default=False,
                ):
                    break
                with console.status("Searching for earliest retirement age (both together)..."):
                    rs = run_retirement_search(
                        inputs=inputs,
                        seed=inputs.seed,
                        mode="both_together",
                        success_threshold=threshold,
                    )
                session.retirement_search_all = rs
            _display_retirement_search_result(rs, main_trials=session.results.trials)
            break

        # Per earner
        console.print()
        console.print(f"  [{THEME_COLOR_BRIGHT}]Which earner to optimise?[/]")
        for i, e in enumerate(earners):
            console.print(
                f"    [{THEME_COLOR}]{i + 1}.[/] {escape(e.label)}"
                f" (currently retires at {e.retirement_age})"
            )
        earner_choice = _prompt_int("  Choice", default=1, lo=1, hi=len(earners))
        target_idx = earner_choice - 1
        target_label = earners[target_idx].label

        rs = session.retirement_search_per_earner.get(target_label)
        if rs is None or rs.mode != "per_earner":
            fixed_info = [
                f"{escape(e.label)} at {e.retirement_age}"
                for i, e in enumerate(earners)
                if i != target_idx
            ]
            console.print(f"  [dim]Others held fixed: {', '.join(fixed_info)}.[/]")
            if not _prompt_yn(
                "Run search? This will run several full simulations and may take 2-3 minutes.",
                default=False,
            ):
                break
            with console.status(
                f"Searching for earliest feasible retirement age ({escape(target_label)})..."
            ):
                rs = run_retirement_search(
                    inputs=inputs,
                    seed=inputs.seed,
                    mode="per_earner",
                    target_earner_index=target_idx,
                    success_threshold=threshold,
                )
            session.retirement_search_per_earner[target_label] = rs

        _display_retirement_search_result(rs, main_trials=session.results.trials)
        if not _prompt_yn("Search for another earner?", default=False):
            break


def _display_retirement_search_result(rs: RetirementSearchResult, main_trials: int) -> None:
    """Render a retirement-age search result, branching on feasibility."""
    target = f"{rs.threshold * 100:.0f}%"
    console.print()
    console.print(
        Panel(
            f"[bold {THEME_COLOR_BRIGHT}]Earliest feasible retirement age[/]",
            border_style=THEME_COLOR_ACCENT,
        )
    )
    if rs.target_earner_label:
        console.print(f"  Searched age: {escape(rs.target_earner_label)} (others held fixed)")
    console.print(
        f"  Your entered retirement age: {rs.entered_age} "
        f"({_fmt_share(rs.entered_p_success)} success)"
    )
    console.print(
        f"  Based on {rs.trials:,} simulated markets; your main result used {main_trials:,}."
    )
    if not rs.entered_plan_feasible:
        console.print(
            f"  [{THEME_COLOR_WARN}]Your entered age does not meet the {target} threshold.[/]"
        )

    if rs.non_monotonic and len(rs.non_monotonic_ages) == 2:
        lower, higher = rs.non_monotonic_ages
        console.print(
            f"  [{THEME_COLOR_WARN}]Results are not monotonic at ages {lower} and {higher}: "
            f"a younger age passed while an older age failed, so the answer is "
            f"provisional.[/]"
        )
    elif rs.non_monotonic:
        console.print(
            f"  [{THEME_COLOR_WARN}]Results are not monotonic in age, so the answer is "
            "provisional.[/]"
        )

    console.print()
    if not rs.feasible or rs.earliest_feasible_age is None:
        console.print(
            f"  [bold {THEME_COLOR_ERROR}]No retirement age from {rs.entered_age} down to "
            f"{rs.floor_age} meets the {target} threshold.[/]"
        )
    elif rs.earliest_feasible_age >= rs.entered_age:
        console.print(
            f"  Your entered age {rs.entered_age} is the earliest age that meets the "
            f"{target} threshold."
        )
    else:
        console.print(
            f"  Earliest age meeting the {target} threshold: "
            f"[bold {THEME_COLOR_BRIGHT}]{rs.earliest_feasible_age}[/] "
            f"({_fmt_share(rs.earliest_p_success)} success)"
        )

    if rs.evaluated_ages:
        console.print()
        age_label = "Age (target earner)" if rs.target_earner_label else "Age"
        table = _table("Ages evaluated")
        table.add_column(age_label, style=THEME_COLOR_BRIGHT, justify="right")
        table.add_column("Success", justify="right", style=THEME_COLOR, no_wrap=True)
        table.add_column("Meets threshold", justify="right", no_wrap=True)
        table.add_column("", style="dim")
        for age in sorted(rs.evaluated_ages, reverse=True):
            p = rs.evaluated_ages[age]
            meets = p >= rs.threshold
            table.add_row(
                str(age),
                _fmt_share(p),
                "yes" if meets else "no",
                "\u2190 entered" if age == rs.entered_age else "",
            )
        console.print(table)

    console.print()
    console.print(
        "  [dim]Note: this search is specific to your current plan configuration. Changing any "
        "input will change this result.[/]"
    )


def _view_mortgage_amortisation(session: ResultsSession) -> None:
    """Median mortgage balance path, with rate and offset columns where they apply."""
    r = session.results
    household = session.household
    if not r.mortgage_by_age and not household.mortgages:
        console.print("  [dim]No mortgage in this scenario.[/]")
        return
    ages = r.mortgage_by_age_ages
    if not r.mortgage_by_age or not ages:
        console.print("  [dim]No mortgage trajectory data available.[/]")
        return

    offset_labels = [a.label for a in household.investment_accounts if a.is_offset]
    any_neutral_marker = False
    for mortgage in household.mortgages:
        mdata = r.mortgage_by_age.get(mortgage.label)
        if mdata is None:
            continue
        p5 = mdata.get("p5", [])
        p50 = mdata["p50"]
        p95 = mdata.get("p95", [])
        n_rows = min(len(ages), len(p50))

        ratedata = r.mortgage_rate_by_age.get(mortgage.label)
        rates: tuple[Sequence[float], Sequence[float], Sequence[float]] | None = None
        if mortgage.interest_rate_stochastic and ratedata is not None and ratedata.get("p50"):
            rates = (ratedata.get("p5", []), ratedata["p50"], ratedata.get("p95", []))

        linked = [label for label in offset_labels if label in mortgage.offset_accounts]
        offset_series = _offset_median_series(r, linked, n_rows) if linked else None

        bal_p5 = [p5[i] if i < len(p5) else p50[i] for i in range(n_rows)]
        bal_p95 = [p95[i] if i < len(p95) else p50[i] for i in range(n_rows)]
        fmt_p5 = _money_formatter(bal_p5)
        fmt_p50 = _money_formatter(p50[:n_rows])
        fmt_p95 = _money_formatter(bal_p95)
        fmt_offset = _money_formatter(offset_series) if offset_series is not None else None

        table = _table(f"Mortgage amortisation: {escape(mortgage.label)} (nominal dollars)")
        table.add_column("Age", style=THEME_COLOR_BRIGHT, justify="right")
        table.add_column("P5", justify="right", style=THEME_COLOR, no_wrap=True)
        table.add_column("Median", justify="right", style=THEME_COLOR_BRIGHT, no_wrap=True)
        table.add_column("P95", justify="right", style=THEME_COLOR, no_wrap=True)
        if rates is not None:
            table.add_column("Rate P5", justify="right", style="dim", no_wrap=True)
            table.add_column("Rate P50", justify="right", style="dim", no_wrap=True)
            table.add_column("Rate P95", justify="right", style="dim", no_wrap=True)
        if offset_series is not None:
            table.add_column(
                "Offset (sum of medians)", justify="right", style=THEME_COLOR, no_wrap=True
            )
        table.add_column("", style="dim")

        for i in range(n_rows):
            bal = p50[i]
            row = [str(ages[i]), fmt_p5(bal_p5[i]), fmt_p50(bal), fmt_p95(bal_p95[i])]
            if rates is not None:
                row.extend(_cell(series, i, _fmt_rate) for series in rates)
            marker = ""
            if offset_series is not None and fmt_offset is not None:
                offset_bal = offset_series[i]
                row.append(fmt_offset(offset_bal))
                if bal > 0 and offset_bal >= bal:
                    marker = "\u2190 neutral"
                    any_neutral_marker = True
            if bal <= 0:
                marker = "\u2713 cleared"
            row.append(marker)
            table.add_row(*row)
        console.print()
        console.print(table)

    console.print()
    console.print(
        "  [dim]Mortgage balances are nominal (then-year) dollars. Offset is the sum of the "
        "median balances of all linked offset accounts, not a median of the total.[/]"
    )
    if any_neutral_marker:
        console.print(
            "  [dim]\u2190 neutral: the offset balance covers the mortgage balance at that "
            "age.[/]"
        )
