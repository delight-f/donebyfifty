"""Tests for the wizard, prompt helpers and results display."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.prompt import Prompt

from donebyfifty.config import DEFAULT_EDU_SCHEDULE
from donebyfifty.models import (
    Child,
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
    SimulationInputs,
    SimulationResults,
)
from donebyfifty.ui_common import _parse_monetary, _prompt_pct, _prompt_yn
from donebyfifty.ui_results import display_results
from donebyfifty.ui_wizard import (
    _configure_child,
    _configure_children_section,
    _configure_earner,
    _configure_mortgage,
    _configure_mortgages_section,
    _display_contribution_warnings,
    _display_earner_super_warnings,
    _link_offsets_to_mortgages,
    _super_warning_messages,
    configure_simulation_params,
    review_before_run,
)


@contextmanager
def _recording_console() -> Iterator[Console]:
    """Route wizard, common and results output into a recording console."""
    rec = Console(record=True, width=200, file=io.StringIO())
    with (
        patch("donebyfifty.ui_common.console", rec),
        patch("donebyfifty.ui_wizard.console", rec),
        patch("donebyfifty.ui_results.console", rec),
    ):
        yield rec


def _scripted(overrides: dict[str, str]) -> Callable[..., str]:
    """Answer a prompt by matching its text; otherwise return its default.

    A prompt with no scripted answer and no default raises, so a test cannot
    hang in the helpers' re-prompt loops.
    """

    def ask(prompt: object, *args: Any, **kwargs: Any) -> str:
        text = str(prompt)
        for key, answer in overrides.items():
            if key in text:
                return answer
        default = kwargs.get("default")
        if default is None:
            raise AssertionError(f"unscripted prompt without default: {text!r}")
        return str(default)

    return ask


# ── _parse_monetary ─────────────────────────────────────────────────────────


class TestParseMonetary:
    def test_shorthand_suffixes(self) -> None:
        assert _parse_monetary("300k") == 300_000.0
        assert _parse_monetary("1.5M") == 1_500_000.0
        assert _parse_monetary("2b") == 2_000_000_000.0

    def test_plain_number(self) -> None:
        assert _parse_monetary("150000") == 150_000.0

    def test_dollar_and_commas(self) -> None:
        assert _parse_monetary("$100,000") == 100_000.0

    def test_percent_sign_is_stripped(self) -> None:
        # The caller decides the unit: 6.5% parses to 6.5.
        assert _parse_monetary("6.5%") == 6.5

    def test_invalid_returns_none(self) -> None:
        assert _parse_monetary("abc") is None
        assert _parse_monetary("") is None

    def test_non_finite_returns_none(self) -> None:
        assert _parse_monetary("inf") is None
        assert _parse_monetary("nan") is None


# ── Prompt helpers ──────────────────────────────────────────────────────────


class TestPromptHelpers:
    def test_yn_reprompts_until_valid(self) -> None:
        with (
            _recording_console() as rec,
            patch.object(Prompt, "ask", side_effect=["maybe", "y"]) as ask,
        ):
            assert _prompt_yn("Proceed?", default=False) is True
        assert ask.call_count == 2
        assert "Please enter y or n" in rec.export_text()

    def test_pct_percent_sign_returns_decimal(self) -> None:
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=["6.5%"]),
        ):
            assert _prompt_pct("Rate [%]") == 0.065

    def test_pct_bare_fraction_asks_for_confirmation(self) -> None:
        # "0.065" is below 1 and has no '%', so it is confirmed as 6.5%.
        answers = {"Did you mean": "y", "Rate": "0.065"}
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted(answers)) as ask,
        ):
            assert _prompt_pct("Rate [%]") == 0.065
        asked = [str(call.args[0]) for call in ask.call_args_list]
        assert any("Did you mean 6.5%?" in text for text in asked)


# ── Super and contribution warnings ─────────────────────────────────────────


class TestSuperWarnings:
    def test_low_income_has_no_warnings(self) -> None:
        # Salary 50,000 + SG 6,000 = 56,000, well below the 250,000 Div 293 threshold.
        assert _super_warning_messages("E", 50_000, 0.12, 13) == []

    def test_sg_exceeds_concessional_cap(self) -> None:
        # Hand calc: min(300,000, 270,830) * 0.20 = 54,166 > 32,500 cap (no growth, years=0).
        msgs = _super_warning_messages("E", 300_000, 0.20, 0, sacrifice=0.0)
        assert any("exceeds concessional cap" in m for m in msgs)

    def test_div293_not_triggered_below_threshold(self) -> None:
        # Hand calc: 220,000 salary + 220,000 * 0.12 SG = 246,400 < 250,000.
        assert _super_warning_messages("E", 220_000, 0.12, 0, sacrifice=0.0) == []
        assert _super_warning_messages("E", 220_000, 0.12, 0, sacrifice=20_000.0) == []

    def test_div293_triggered_above_threshold_regardless_of_sacrifice(self) -> None:
        # Hand calc: 240,000 + 240,000 * 0.12 = 268,800 > 250,000. Salary sacrifice
        # moves income between the taxable and the super lines and cancels out.
        for sacrifice in (0.0, 20_000.0):
            msgs = _super_warning_messages("E", 240_000, 0.12, 0, sacrifice=sacrifice)
            assert any("Division 293" in m for m in msgs), sacrifice

    def test_display_earner_warning_panel_renders(self) -> None:
        with _recording_console() as rec:
            _display_earner_super_warnings("E", 240_000, 0.12, 50, 37)
        assert "Division 293" in rec.export_text()

    def test_contribution_warnings_silent_for_modest_household(self) -> None:
        hh = Household(earners=(Earner(salary=100_000, sg_rate=0.12, super_access_age=60),))
        with _recording_console() as rec:
            _display_contribution_warnings(hh, 37)
        assert "Division 293" not in rec.export_text()
        assert "exceeds concessional cap" not in rec.export_text()

    def test_contribution_warnings_flag_high_income(self) -> None:
        hh = Household(
            earners=(Earner(salary=240_000, sg_rate=0.12, retirement_age=50, super_access_age=60),)
        )
        with _recording_console() as rec:
            _display_contribution_warnings(hh, 37)
        assert "Division 293" in rec.export_text()


# ── Wizard prompt flows ─────────────────────────────────────────────────────


class TestWizardFlows:
    def test_no_mortgages_answer_skips_count_prompt(self) -> None:
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({"Any mortgages": "n"})) as ask,
        ):
            assert _configure_mortgages_section(None) == ()
        # Only the yes/no question was asked; no count prompt followed.
        assert ask.call_count == 1

    def test_no_children_custom_answer_skips_count_prompt(self) -> None:
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({"children": "n"})) as ask,
        ):
            assert _configure_children_section("custom", None) == ()
        assert ask.call_count == 1

    def test_interest_only_mortgage_stores_zero_payment(self) -> None:
        answers = {
            "Remaining balance": "400000",
            "Interest rate": "6",
            "Interest-only loan": "y",
        }
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted(answers)),
        ):
            m = _configure_mortgage("Mortgage 1")
        # The engine reads interest-only as a zero monthly payment.
        assert m.principal == 400_000.0
        assert m.interest_rate == pytest.approx(0.06)
        assert m.monthly_payment == 0.0

    def test_pi_mortgage_keeps_entered_payment(self) -> None:
        # Hand calc: 400,000 * 6% = 24,000 interest a year; 3,000 * 12 = 36,000 covers it,
        # so no interest-only warning is printed.
        answers = {
            "Remaining balance": "400000",
            "Interest rate": "6",
            "Interest-only loan": "n",
            "Monthly payment": "3000",
        }
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted(answers)),
        ):
            m = _configure_mortgage("Mortgage 1")
        assert m.monthly_payment == 3_000.0

    def test_offset_link_escapes_markup_and_is_idempotent(self) -> None:
        offset = InvestmentAccount(label="Offset", is_offset=True)
        mortgages = [
            MortgageAccount(label="Mortgage 1"),
            MortgageAccount(label="Mortgage [bold]x"),
        ]
        with (
            _recording_console() as rec,
            patch.object(Prompt, "ask", side_effect=_scripted({"Choice": "2"})),
        ):
            linked = _link_offsets_to_mortgages([offset], mortgages)
        # Markup in the label is printed literally, not interpreted.
        assert "Mortgage [bold]x" in rec.export_text()
        assert linked[0].offset_accounts == ()
        assert linked[1].offset_accounts == ("Offset",)

        # Re-running on already-linked mortgages must not prompt again.
        with patch.object(Prompt, "ask") as ask:
            again = _link_offsets_to_mortgages([offset], list(linked))
        ask.assert_not_called()
        assert again[1].offset_accounts == ("Offset",)

    def test_earner_glide_defaults_to_fifteen_years(self) -> None:
        # defaults=None means no glide yet, so the default duration is 15.
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({"Enable age-based glidepath": "y"})),
        ):
            e = _configure_earner("Earner 1", None, start_age=37)
        assert e.super_glide_end_year == 15
        assert e.super_glide_target_pct == 30.0

    def test_simulation_params_keep_passed_household(self) -> None:
        # H10: the household passed in must be the one returned, not a rebuilt default.
        hh = Household(earners=(Earner(label="Earner 1", salary=123_000, super_access_age=60),))
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({})),
        ):
            sim = configure_simulation_params(existing=None, household=hh, start_age=37)
        assert sim.household is hh


# ── Education costs ─────────────────────────────────────────────────────────


def _prompt_default(ask: Any, text: str) -> object:
    """Return the default passed to the first prompt whose text contains ``text``."""
    for call in ask.call_args_list:
        if text in str(call.args[0]):
            return call.kwargs.get("default")
    raise AssertionError(f"no prompt containing {text!r}")


class TestEducationCosts:
    def test_new_child_defaults_to_default_schedule(self) -> None:
        # No existing child: the default choice is 1, the config table (ages 0-17, 18 rows).
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({})) as ask,
        ):
            child = _configure_child("Child 1")
        assert _prompt_default(ask, "Choice") == "1"
        assert child.education_schedule == DEFAULT_EDU_SCHEDULE
        assert [age for age, _ in child.education_schedule] == list(range(18))

    def test_no_education_costs_stores_empty_schedule(self) -> None:
        # Choice 3 must store (), which the engine reads as no costs.
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({"Choice": "3"})),
        ):
            child = _configure_child("Child 1")
        assert child.education_schedule == ()

    def test_custom_costs_store_entered_pairs(self) -> None:
        # Prompts in order: child age 5, choice 2, pair age 6, cost 6000, blank age to finish.
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=["5", "2", "6", "6000", ""]),
        ):
            child = _configure_child("Child 1")
        assert child.education_schedule == ((6, 6000.0),)

    def test_custom_choice_with_no_pairs_means_no_costs(self) -> None:
        # Prompts in order: child age 5, choice 2, then a blank age straight away.
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=["5", "2", ""]),
        ):
            child = _configure_child("Child 1")
        assert child.education_schedule == ()

    def test_existing_default_schedule_defaults_choice_to_one(self) -> None:
        existing = Child(label="Child 1", age=5, education_schedule=DEFAULT_EDU_SCHEDULE)
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({})) as ask,
        ):
            child = _configure_child("Child 1", existing)
        assert _prompt_default(ask, "Choice") == "1"
        assert child.education_schedule == DEFAULT_EDU_SCHEDULE

    def test_existing_empty_schedule_defaults_choice_to_no_costs(self) -> None:
        existing = Child(label="Child 1", age=5, education_schedule=())
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({})) as ask,
        ):
            child = _configure_child("Child 1", existing)
        assert _prompt_default(ask, "Choice") == "3"
        assert child.education_schedule == ()

    def test_existing_custom_schedule_defaults_choice_to_custom(self) -> None:
        existing = Child(label="Child 1", age=5, education_schedule=((6, 6000.0),))
        with (
            _recording_console(),
            patch.object(Prompt, "ask", side_effect=_scripted({})) as ask,
        ):
            _configure_child("Child 1", existing)
        assert _prompt_default(ask, "Choice") == "2"

    def test_review_labels_each_education_choice(self) -> None:
        # Real Console output: the review screen must name each child's education choice.
        hh = Household(
            earners=(Earner(label="Earner 1", super_access_age=60),),
            children=(
                Child(label="Kid A", age=3, education_schedule=DEFAULT_EDU_SCHEDULE),
                Child(label="Kid B", age=4, education_schedule=()),
                Child(label="Kid C", age=5, education_schedule=((6, 6000.0),)),
            ),
        )
        with (
            _recording_console() as rec,
            patch.object(Prompt, "ask", side_effect=_scripted({"Choice": "4"})),
        ):
            action = review_before_run(hh, SimulationInputs(household=hh))
        assert action == "cancel"
        text = rec.export_text()
        assert "Kid A: age 3, default schedule" in text
        assert "Kid B: age 4, no education costs" in text
        assert "Kid C: age 5, custom schedule" in text


# ── Results display ─────────────────────────────────────────────────────────


def _results(p_success: float) -> SimulationResults:
    """Minimal SimulationResults with the fields the display reads."""
    return SimulationResults(
        trials=1000,
        p_success=p_success,
        bridge_mean=400_000.0,
        bridge_median=390_000.0,
        bridge_p5=100_000.0,
        bridge_p10=150_000.0,
        bridge_p25=250_000.0,
        bridge_p75=500_000.0,
        bridge_p90=650_000.0,
        bridge_p95=750_000.0,
        bridge_min=0.0,
        super_median=600_000.0,
        horizon_age=60,
        per_earner_super_p50={"Earner 1": 600_000.0},
        remaining_mortgage_p50={},
    )


class TestDisplayResults:
    def test_high_risk_tier_and_threshold_status(self) -> None:
        # 40% falls in the High risk tier (30-50%) and misses the 95% default threshold.
        with _recording_console() as rec:
            display_results(_results(0.40), start_age=37)
        text = rec.export_text()
        assert "High risk" in text
        assert "(threshold 95%: not reached)" in text

    def test_severe_risk_below_thirty_percent(self) -> None:
        with _recording_console() as rec:
            display_results(_results(0.20), start_age=37)
        assert "Severe risk" in rec.export_text()

    def test_meeting_threshold_has_no_risk_warning(self) -> None:
        with _recording_console() as rec:
            display_results(_results(0.97), start_age=37)
        text = rec.export_text()
        assert "(threshold 95%: reached)" in text
        assert "High risk" not in text
        assert "Severe risk" not in text
        assert "is below the" not in text
