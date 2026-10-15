"""Behavioural tests for the results display (ui_results).

Each test renders through a recording console and asserts on the printed text.
Rich wraps table titles at narrow widths, so text checks compare whitespace-normalised output.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from rich.console import Console

from donebyfifty.analysis import RetirementSearchResult, ScenarioComparisonResult
from donebyfifty.models import (
    Household,
    MortgageAccount,
    ResultsSession,
    SimulationInputs,
    SimulationResults,
)
from donebyfifty.ui_results import (
    _display_retirement_search_result,
    _view_drawdown_composition,
    _view_mortgage_amortisation,
    _view_scenario_comparison,
    display_results,
)


@contextmanager
def _recording_console(width: int = 200) -> Iterator[Console]:
    rec = Console(record=True, width=width, file=io.StringIO())
    with patch("donebyfifty.ui_results.console", rec):
        yield rec


def _flat(text: str) -> str:
    """Collapse all whitespace so wrapped table titles compare as one line."""
    return " ".join(text.split())


def _results(p_success: float, trials: int = 1000, **overrides: Any) -> SimulationResults:
    """Minimal SimulationResults with the fields the display reads."""
    base: dict[str, Any] = {
        "trials": trials,
        "p_success": p_success,
        "bridge_mean": 400_000.0,
        "bridge_median": 390_000.0,
        "bridge_p5": 100_000.0,
        "bridge_p10": 150_000.0,
        "bridge_p25": 250_000.0,
        "bridge_p75": 500_000.0,
        "bridge_p90": 650_000.0,
        "bridge_p95": 750_000.0,
        "bridge_min": 0.0,
        "super_median": 600_000.0,
        "horizon_age": 60,
        "per_earner_super_p50": {"Earner 1": 600_000.0},
        "remaining_mortgage_p50": {},
    }
    base.update(overrides)
    return SimulationResults(**base)


def _search_result(**overrides: Any) -> RetirementSearchResult:
    """Infeasible search by default: no passing age, sentinel ``earliest_age`` = floor - 1."""
    base: dict[str, Any] = {
        "entered_age": 50,
        "entered_p_success": 0.50,
        "earliest_age": 39,
        "earliest_p_success": 0.0,
        "floor_age": 40,
        "threshold": 0.95,
        "mode": "both_together",
        "target_earner_label": None,
        "entered_ages_by_earner": {"Earner 1": 50},
        "entered_plan_feasible": False,
        "feasible": False,
        "earliest_feasible_age": None,
        "non_monotonic": False,
        "non_monotonic_ages": [],
        "evaluated_ages": {40: 0.10, 50: 0.50},
        "trials": 10_000,
    }
    base.update(overrides)
    return RetirementSearchResult(**base)


class TestSuccessHeadline:
    def test_near_total_success_is_not_rounded_to_100(self) -> None:
        # 1 - 0.99998 = 0.00002; 0.00002 * 100,000 trials = 2 failed paths.
        # 99.998% would round to "100.0%" at one decimal, so it must print as "≥99.9%".
        with _recording_console() as rec:
            display_results(_results(0.99998, trials=100_000), start_age=37)
        text = _flat(rec.export_text())
        assert "Success probability: ≥99.9%" in text
        assert "(2 of 100,000 paths failed)" in text


class TestNotCheckableMortgage:
    def test_message_prints_trial_count_not_a_reason(self) -> None:
        # Mapping is label -> count of trials; 100 of 100 trials are uncheckable.
        results = _results(0.9, trials=100, per_mortgage_not_checkable={"Home": 100})
        with _recording_console() as rec:
            display_results(results, start_age=37)
        text = _flat(rec.export_text())
        assert "Home: term end is beyond the simulation horizon (age 60)" in text
        assert "100 of 100 trials" in text


class TestSuperLabels:
    def test_super_figures_are_labelled_todays_dollars(self) -> None:
        with _recording_console() as rec:
            display_results(_results(0.9), start_age=37)
        text = _flat(rec.export_text())
        assert "Median super balance at end of scenario (today's dollars)" in text
        assert "Median Super per Earner (today's dollars)" in text


class TestRetirementSearch:
    def test_infeasible_search_prints_no_placeholder_age(self) -> None:
        # Sentinel earliest_age = floor_age - 1 = 39 must never be shown as an answer.
        rs = _search_result()
        with _recording_console() as rec:
            _display_retirement_search_result(rs, main_trials=5_000)
        text = _flat(rec.export_text())
        assert "No retirement age from 50 down to 40 meets the 95% threshold." in text
        assert "39" not in text

    def test_feasible_search_reports_earliest_age(self) -> None:
        rs = _search_result(
            feasible=True,
            earliest_age=44,
            earliest_feasible_age=44,
            earliest_p_success=0.96,
            evaluated_ages={44: 0.96, 50: 0.99},
            entered_p_success=0.99,
            entered_plan_feasible=True,
        )
        with _recording_console() as rec:
            _display_retirement_search_result(rs, main_trials=5_000)
        text = _flat(rec.export_text())
        assert "Earliest age meeting the 95% threshold: 44" in text
        assert "(96.0% success)" in text
        assert "Based on 10,000 simulated markets; your main result used 5,000." in text


class TestScenarioComparison:
    def test_change_is_shown_in_percentage_points_with_half_width(self) -> None:
        # Base p = 0.95. delta_p = -0.05 -> -5.0 pp. delta_se = 0.01 -> half-width
        # 1.96 * 0.01 * 100 = 1.96 -> "2.0". |-5.0| >= 1.96, so no "within noise" flag.
        base = _results(0.95)
        scen = ScenarioComparisonResult(
            label="Delay retirement by 1 year",
            p_success=0.90,
            bridge_p5=100_000.0,
            bridge_median=400_000.0,
            delta_p=-0.05,
            delta_se=0.01,
        )
        household = Household()
        session = ResultsSession(
            results=base,
            household=household,
            inputs=SimulationInputs(household=household),
            scenarios={scen.label: scen},
        )
        with _recording_console() as rec:
            _view_scenario_comparison(session)
        text = _flat(rec.export_text())
        assert "Delay retirement by 1 year" in text
        assert "-5.0 pp \u00b1 2.0" in text
        assert "within noise" not in text
        assert "pp = percentage points" in text

    def test_long_scenario_label_does_not_wrap_at_80_columns(self) -> None:
        # A wide change header used to squeeze the Scenario column and wrap this label.
        label = "Earner 1 stops working now (age 37)"
        scen = ScenarioComparisonResult(
            label=label,
            p_success=0.0,
            bridge_p5=-1_080_000.0,
            bridge_median=0.0,
            delta_p=-0.70,
            delta_se=0.0045,
        )
        household = Household()
        session = ResultsSession(
            results=_results(0.694),
            household=household,
            inputs=SimulationInputs(household=household),
            scenarios={label: scen},
        )
        with _recording_console(width=80) as rec:
            _view_scenario_comparison(session)
        assert label in rec.export_text()


class TestMortgageAmortisation:
    def _session(self, label: str) -> ResultsSession:
        mortgage = MortgageAccount(
            label=label,
            principal=500_000.0,
            interest_rate=0.0605,
            monthly_payment=3_000.0,
        )
        household = Household(mortgages=(mortgage,))
        results = _results(
            0.9,
            mortgage_by_age_ages=[37, 38],
            mortgage_by_age={
                label: {
                    "p5": [480_000.0, 470_000.0],
                    "p50": [500_000.0, 490_000.0],
                    "p95": [520_000.0, 510_000.0],
                }
            },
        )
        return ResultsSession(
            results=results,
            household=household,
            inputs=SimulationInputs(household=household),
        )

    def test_fixed_rate_table_has_no_truncation_at_80_columns(self) -> None:
        session = self._session("Home")
        with _recording_console(width=80) as rec:
            _view_mortgage_amortisation(session)
        text = rec.export_text()
        assert "\u2026" not in text
        assert "Mortgage amortisation: Home (nominal dollars)" in _flat(text)

    def test_label_with_markup_brackets_renders_literally(self) -> None:
        session = self._session("Home [/] loan")
        with _recording_console() as rec:
            _view_mortgage_amortisation(session)
        assert "Home [/] loan" in _flat(rec.export_text())


class TestDrawdownComposition:
    def test_headings_and_headers_do_not_wrap_at_80_columns(self) -> None:
        results = _results(
            0.694,
            bridge_by_age_ages=[37, 38, 39, 40],
            offset_drawn_p50=[12_000.0, 11_000.0, 9_000.0, 0.0],
            non_offset_drawn_p50=[0.0, 3_000.0, 5_000.0, 40_000.0],
            cgt_paid_p50=[0.0, 1_200.0, 2_500.0, 0.0],
            income_tax_p50=[0.0, -8_836.0, 2_100.0, 0.0],
            income_tax_total_p5=-54_004.0,
            income_tax_total_p50=-20_000.0,
            income_tax_total_p95=3_000.0,
        )
        household = Household()
        session = ResultsSession(
            results=results,
            household=household,
            inputs=SimulationInputs(household=household),
        )
        with _recording_console(width=80) as rec:
            _view_drawdown_composition(session)
        text = rec.export_text()
        assert "Drawdown source composition (median per year, today's dollars)" in text
        assert "Investment-income tax (median per year, today's dollars)" in text
        assert any("Age" in line and "Non-offset drawn" in line for line in text.splitlines())
