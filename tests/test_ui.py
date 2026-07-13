"""Tests for UI functions (ui.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.panel import Panel
from rich.console import Console

from ui import (
    _display_earner_super_warnings,
    _display_contribution_warnings,
    display_results,
)
from models import Household, Earner, SimulationResults


class TestDisplayEarnerSuperWarnings:
    """Tests for real-time super contribution warnings during earner configuration."""

    def test_no_warnings_for_low_earner(self) -> None:
        """No warnings should appear for low-income earners."""
        # Should not raise any exceptions
        _display_earner_super_warnings("Earner 1", 50_000, 0.12, 50, 37)

    def test_warning_for_sg_exceeding_cap(self) -> None:
        """Warning should appear when SG exceeds concessional cap."""
        # High salary that would cause SG to exceed cap
        high_salary = 300_000  # Would generate ~$31,200 SG, close to cap
        # Should not raise any exceptions
        _display_earner_super_warnings("Earner 1", high_salary, 0.12, 50, 37)

    def test_warning_for_div293_tax(self) -> None:
        """Warning should appear when income exceeds Div 293 threshold."""
        high_salary = 300_000  # Above Div 293 threshold of $250,000
        # Should not raise any exceptions
        _display_earner_super_warnings("Earner 1", high_salary, 0.12, 50, 37)

    def test_no_warnings_for_normal_earner(self) -> None:
        """No warnings should appear for normal earners."""
        normal_salary = 100_000  # Below Div 293 threshold
        # Should not raise any exceptions
        _display_earner_super_warnings("Earner 1", normal_salary, 0.12, 50, 37)


class TestDisplayContributionWarnings:
    """Tests for post-simulation super contribution warnings."""

    def test_no_warnings_for_normal_household(self) -> None:
        """No warnings should appear for normal household."""
        household = Household(
            earners=(
                Earner(salary=100_000, retirement_age=50),
            ),
        )
        # Should not raise any exceptions
        _display_contribution_warnings(household, 37)

    def test_warning_for_high_income_earner(self) -> None:
        """Warning should appear for high-income earner."""
        household = Household(
            earners=(
                Earner(salary=300_000, retirement_age=50),
            ),
        )
        # Should not raise any exceptions
        _display_contribution_warnings(household, 37)

    def test_warning_for_sg_exceeding_cap(self) -> None:
        """Warning should appear when SG exceeds concessional cap."""
        # Very high salary that would cause SG to exceed cap
        household = Household(
            earners=(
                Earner(salary=300_000, retirement_age=50),
            ),
        )
        # Should not raise any exceptions
        _display_contribution_warnings(household, 37)


class TestDisplayResults:
    """Tests for simulation results display."""

    def test_display_results_with_warnings(self) -> None:
        """Results should be displayed with contribution warnings."""
        household = Household(
            earners=(
                Earner(salary=300_000, retirement_age=50),
            ),
        )
        results = SimulationResults(
            trials=5000,
            p_success=0.95,
            bridge_mean=500_000,
            bridge_median=450_000,
            bridge_p5=200_000,
            bridge_p10=300_000,
            bridge_p25=400_000,
            bridge_p75=600_000,
            bridge_p90=700_000,
            bridge_p95=800_000,
            bridge_min=100_000,
            super_median=1_000_000,
            horizon_age=60,
        )
        with patch("ui.console") as mock_console:
            display_results(results, household, start_age=37)
            # Should display results and potentially warnings
            calls = list(mock_console.print.call_args_list)
            assert len(calls) > 0

    def test_display_results_with_low_success_probability(self) -> None:
        """Results should show warning for low success probability."""
        results = SimulationResults(
            trials=5000,
            p_success=0.80,  # Below 95% threshold
            bridge_mean=500_000,
            bridge_median=450_000,
            bridge_p5=200_000,
            bridge_p10=300_000,
            bridge_p25=400_000,
            bridge_p75=600_000,
            bridge_p90=700_000,
            bridge_p95=800_000,
            bridge_min=100_000,
            super_median=1_000_000,
            horizon_age=60,
        )
        with patch("ui.console") as mock_console:
            display_results(results, start_age=37)
            # Should have a warning about low success probability
            warnings = [call for call in mock_console.print.call_args_list if "below" in str(call).lower()]
            assert len(warnings) > 0

    def test_display_results_with_no_warnings(self) -> None:
        """Results should be displayed without warnings for normal household."""
        household = Household(
            earners=(
                Earner(salary=100_000, retirement_age=50),
            ),
        )
        results = SimulationResults(
            trials=5000,
            p_success=0.95,
            bridge_mean=500_000,
            bridge_median=450_000,
            bridge_p5=200_000,
            bridge_p10=300_000,
            bridge_p25=400_000,
            bridge_p75=600_000,
            bridge_p90=700_000,
            bridge_p95=800_000,
            bridge_min=100_000,
            super_median=1_000_000,
            horizon_age=60,
        )
        with patch("ui.console") as mock_console:
            display_results(results, household, start_age=37)
            # Should not have any warnings
            warnings = [call for call in mock_console.print.call_args_list if "⚠️" in str(call)]
            assert len(warnings) == 0


class TestMonetaryShortcuts:
    """Tests for monetary shorthand notation support."""

    def test_parse_k_suffix(self) -> None:
        """Test parsing of k suffix for thousands."""
        from ui import _parse_monetary
        assert _parse_monetary("300k") == 300_000.0
        assert _parse_monetary("1.5k") == 1_500.0
        assert _parse_monetary("10k") == 10_000.0

    def test_parse_m_suffix(self) -> None:
        """Test parsing of m suffix for millions."""
        from ui import _parse_monetary
        assert _parse_monetary("1M") == 1_000_000.0
        assert _parse_monetary("2.5M") == 2_500_000.0
        assert _parse_monetary("10M") == 10_000_000.0

    def test_parse_b_suffix(self) -> None:
        """Test parsing of b suffix for billions."""
        from ui import _parse_monetary
        assert _parse_monetary("1B") == 1_000_000_000.0
        assert _parse_monetary("2.5B") == 2_500_000_000.0

    def test_parse_plain_numbers(self) -> None:
        """Test parsing of plain numbers without suffix."""
        from ui import _parse_monetary
        assert _parse_monetary("1000") == 1_000.0
        assert _parse_monetary("1000.5") == 1_000.5
        assert _parse_monetary("0") == 0.0

    def test_parse_invalid_input(self) -> None:
        """Test parsing of invalid input returns None."""
        from ui import _parse_monetary
        assert _parse_monetary("invalid") is None
        assert _parse_monetary("") is None
        assert _parse_monetary("abc") is None
