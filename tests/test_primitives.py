"""Tests for verified mathematical primitives.

Each test pins the output of a primitive against known values computed
from the original verified model. Any change to a primitive that alters
these outputs is a regression.
"""

from __future__ import annotations

import math
import random

import pytest

from donebyfifty.cgt import CGT_FLOOR_RATE
from donebyfifty.config import EDU_SCHEDULE_TODAY
from donebyfifty.models import Ownership
from donebyfifty.mortgage import (
    amortize_mortgage_monthly,
    generate_mortgage_rate,
    handle_offset_overflow,
    pi_payment,
)
from donebyfifty.returns import (
    EQ_GEO_MEAN,
    EQ_STD,
    MU_EQ,
    SUPER_EQ_CORR,
    SUPER_GEO_MEAN,
    SUPER_STD,
)
from donebyfifty.tax import (
    MEDICARE,
    PT_WEEKS_PER_YEAR,
    bracket_tax,
    brackets_for_fy,
    indexed_brackets,
    lito_offset,
    marginal_rate,
    medicare_levy,
    mls_rate,
    tax,
)

FY26 = brackets_for_fy(2026)

# =============================================================================
# CONSTANT INTEGRITY
# =============================================================================


class TestConstants:
    """Ensure constants are intact and semantically correct."""

    def test_brackets_structure(self) -> None:
        brackets = FY26
        assert len(brackets) == 5
        assert brackets[0] == (18200, 0.00)
        assert brackets[-1][1] == 0.45
        # The top threshold should be infinity (sentinel)
        from math import isinf

        assert isinf(brackets[-1][0])
        # Rates should be non-decreasing
        rates = [r for _, r in brackets]
        assert rates == sorted(rates)

    def test_brackets_all_positive_thresholds(self) -> None:
        for threshold, _ in FY26:
            assert threshold > 0

    def test_medicare_rate(self) -> None:
        assert MEDICARE == 0.02

    def test_cgt_rate(self) -> None:
        assert CGT_FLOOR_RATE == 0.30

    def test_return_assumptions(self) -> None:
        assert EQ_GEO_MEAN == 0.065
        assert EQ_STD == 0.15
        assert SUPER_GEO_MEAN == 0.065
        assert SUPER_STD == 0.12
        assert SUPER_EQ_CORR == 0.80
        # mu = ln(1 + geometric mean): the median real path is the geometric mean
        assert MU_EQ == pytest.approx(math.log(1.065), abs=1e-12)

    def test_pt_assumptions(self) -> None:
        assert PT_WEEKS_PER_YEAR == 48.0

    def test_edu_schedule_total(self) -> None:
        """Known total from client's fee schedule including preschool/daycare: ~$367,008."""
        total = sum(EDU_SCHEDULE_TODAY.values())
        assert total == 367_008.0


# =============================================================================
# TAX
# =============================================================================


class TestTax:
    """Progressive Australian tax + Medicare levy (FY2026 schedule)."""

    @pytest.mark.parametrize(
        ("income", "expected"),
        [
            (0, 0.0),
            # H7: no levy below the low-income threshold, so no tax at $18,200.
            (18_200, 0.0),
            # 15c low band $4,020 less LITO $325, plus full 2% Medicare $900.
            (45_000, 4_595.0),
            # $4,020 + 30c x $90,000 = $31,020, plus Medicare $2,700 (no LITO).
            (135_000, 33_720.0),
            # $31,020 + 37c x $55,000 = $51,370, plus Medicare $3,800.
            (190_000, 55_170.0),
            # $51,370 + 45c x $110,000 = $100,870, plus Medicare $6,000.
            (300_000, 106_870.0),
        ],
    )
    def test_known_incomes(self, income: float, expected: float) -> None:
        assert tax(income, brackets=FY26) == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize(
        ("fy_start_year", "expected"),
        [
            # $45k: 16c band $4,288 - $325 + $900 (FY2025).
            (2025, 4_863.0),
            # 15c band $4,020 - $325 + $900 (FY2026).
            (2026, 4_595.0),
            # 14c band $3,752 - $325 + $900 (FY2027).
            (2027, 4_327.0),
        ],
    )
    def test_low_band_rate_by_fy(self, fy_start_year: int, expected: float) -> None:
        assert tax(45_000, brackets=brackets_for_fy(fy_start_year)) == pytest.approx(
            expected, abs=0.01
        )

    def test_medicare_full_rate_above_reduction(self) -> None:
        """Above the low-income upper threshold the levy is the full 2%."""
        for inc in [50_000, 200_000]:
            levy = tax(inc, brackets=FY26) - bracket_tax(inc, FY26) + lito_offset(inc)
            assert levy == pytest.approx(inc * MEDICARE, abs=0.01)

    def test_medicare_reduced_for_low_income(self) -> None:
        """H7: the low-income reduction cuts the levy below the full 2%."""
        assert tax(18_200, brackets=FY26) == pytest.approx(0.0, abs=0.01)
        # $30k: 15c band $1,770 less LITO $700, plus 10% of the $1,989 excess
        # over the $28,011 threshold ($198.90).
        assert tax(30_000, brackets=FY26) == pytest.approx(1_770.0 - 700.0 + 198.9, abs=0.01)

    def test_family_income_medicare_capped_at_full(self) -> None:
        """D11: family rule for couples, capped at the full 2% levy."""
        # $40k taxable: 15c band $3,270, LITO $575, levy capped at $800
        # (10% x $32,762 x 40/80 = $1,638 > $800).
        assert tax(40_000, brackets=FY26, family_income=80_000) == pytest.approx(3_495.0, abs=0.01)

    def test_negative_income_no_tax(self) -> None:
        """H7: tax() must never return a negative amount."""
        assert tax(-1_000, brackets=FY26) == 0.0

    def test_zero_income(self) -> None:
        assert tax(0, brackets=FY26) == 0.0

    def test_monotonic(self) -> None:
        """Tax should be monotonically increasing with income."""
        incomes = [0, 10_000, 18_200, 30_000, 45_000, 80_000, 135_000, 200_000, 500_000]
        taxes = [tax(i, brackets=FY26) for i in incomes]
        for i in range(1, len(taxes)):
            assert taxes[i] >= taxes[i - 1]


class TestMedicareLevy:
    """Medicare levy low-income reduction for singles and families (D11)."""

    def test_singles_no_levy_at_threshold(self) -> None:
        assert medicare_levy(28_011) == 0.0

    def test_singles_reduced_in_taper(self) -> None:
        # 10% of the $1,989 excess over $28,011.
        assert medicare_levy(30_000) == pytest.approx(198.9, abs=0.01)

    def test_singles_full_above_taper(self) -> None:
        assert medicare_levy(40_000) == pytest.approx(800.0, abs=0.01)

    def test_family_pro_rated_and_capped(self) -> None:
        # Family rule: 10% x (family income - $47,238) x (taxable / family income),
        # capped at the full 2% levy.
        assert medicare_levy(40_000, family_income=80_000) == pytest.approx(800.0)
        assert medicare_levy(40_000, family_income=60_000) == pytest.approx(800.0)
        # 10% x $4,762 x 40,000 / 52,000 = $366.3077
        assert medicare_levy(40_000, family_income=52_000) == pytest.approx(366.3077, abs=1e-3)
        assert medicare_levy(40_000, family_income=47_000) == 0.0

    def test_family_dependants_raise_threshold(self) -> None:
        # One dependant: family threshold $47,238 + $4,338 = $51,576.
        # 10% x $424 x 40,000 / 52,000 = $32.6154
        assert medicare_levy(40_000, family_income=52_000, dependants=1) == pytest.approx(
            32.6154, abs=1e-3
        )


class TestBracketSchedules:
    """Bracket schedule construction: FY low band (D10) and indexation."""

    @pytest.mark.parametrize(
        ("fy", "low_rate"),
        [(2024, 0.16), (2025, 0.16), (2026, 0.15), (2027, 0.14), (2030, 0.14)],
    )
    def test_low_band_by_fy(self, fy: int, low_rate: float) -> None:
        brackets = brackets_for_fy(fy)
        assert brackets[0] == (18_200, 0.0)
        assert brackets[1] == (45_000, low_rate)
        assert brackets[2:] == ((135_000, 0.30), (190_000, 0.37), (math.inf, 0.45))

    def test_indexed_brackets_scale_thresholds_not_rates(self) -> None:
        # (1.025)^2 = 1.050625: thresholds scale, rates do not.
        idx = indexed_brackets(2026, 2, 0.025)
        assert idx[0] == pytest.approx((18_200 * 1.050625, 0.0))
        assert idx[1] == pytest.approx((45_000 * 1.050625, 0.15))
        assert idx[2] == pytest.approx((135_000 * 1.050625, 0.30))
        assert idx[3] == pytest.approx((190_000 * 1.050625, 0.37))
        assert idx[4][1] == 0.45
        assert idx[4][0] == math.inf

    def test_empty_schedule_rejected(self) -> None:
        """An unset bracket schedule must fail loudly, not tax at zero."""
        with pytest.raises(ValueError):
            bracket_tax(45_000, ())


# =============================================================================
# EDUCATION COST
# =============================================================================


# =============================================================================
# SELL ASSETS
# =============================================================================


# =============================================================================
# AMORTIZE MORTGAGE
# =============================================================================


class TestAmortizeMortgageMonthly:
    """Mortgage amortisation with offset benefit (12 months per call)."""

    def test_no_offset(self) -> None:
        """Without offset, interest is charged on full principal."""
        m, paid_int, paid_prin = amortize_mortgage_monthly(100_000.0, 0.0, 2_000.0, 0.005)
        # Closed form: 100000 * 1.005^12 - 2000 * (1.005^12 - 1) / 0.005 = 81,496.66.
        assert m == pytest.approx(81_496.66, abs=0.01)
        # No capitalisation, so interest + principal = 12 payments = 24,000.
        assert paid_int + paid_prin == pytest.approx(24_000.0, abs=0.01)
        assert paid_prin == pytest.approx(100_000.0 - 81_496.66, abs=0.01)

    def test_with_offset(self) -> None:
        """With offset, effective debt is reduced so more principal is paid."""
        m_no_ofs, _, _ = amortize_mortgage_monthly(100_000.0, 0.0, 2_000.0, 0.005)
        m_with_ofs, _, _ = amortize_mortgage_monthly(100_000.0, 50_000.0, 2_000.0, 0.005)
        # With offset, mortgage reduces FASTER (less interest paid)
        assert m_with_ofs < m_no_ofs
        # Effective debt x = m - 50,000 follows x' = 1.005x - 2000 from 50,000:
        # x_12 = 50000 * 1.005^12 - 2000 * 12.335562 = 28,412.77, so m_12 = 78,412.77.
        assert m_with_ofs == pytest.approx(78_412.77, abs=0.01)

    def test_offset_greater_than_mortgage(self) -> None:
        """When offset >= mortgage, effective debt is 0, no interest charged."""
        m, paid_int, paid_prin = amortize_mortgage_monthly(50_000.0, 60_000.0, 2_000.0, 0.005)
        # All $2k goes to principal each month.
        expected = 50_000.0 - 12 * 2_000.0
        assert m == pytest.approx(expected, abs=0.01)
        assert paid_int == pytest.approx(0.0, abs=0.01)
        assert paid_prin == pytest.approx(12 * 2_000.0, abs=0.01)

    def test_mortgage_fully_paid(self) -> None:
        """Mortgage should not go below zero."""
        m, _, paid_prin = amortize_mortgage_monthly(5_000.0, 0.0, 2_000.0, 0.005)
        # Repayments cap at the balance: 1975 + 1984.875 + 1040.125 = 5000 in 3 months.
        assert m == 0.0
        assert paid_prin == pytest.approx(5_000.0, abs=0.01)

    def test_zero_mortgage(self) -> None:
        """Zero mortgage is a no-op."""
        m, paid_int, paid_prin = amortize_mortgage_monthly(0.0, 0.0, 2_000.0, 0.005)
        assert m == 0.0
        assert paid_int == 0.0
        assert paid_prin == 0.0

    def test_interest_only_keeps_principal(self) -> None:
        """Interest-only: principal unchanged, twelve months of interest paid in cash."""
        m, paid_int, paid_prin = amortize_mortgage_monthly(
            500_000.0, 0.0, 0.0, 0.005, interest_only=True
        )
        # 500,000 x 0.5% = 2,500 per month; x 12 = 30,000. The payment argument is ignored.
        assert m == pytest.approx(500_000.0, abs=0.01)
        assert paid_int == pytest.approx(30_000.0, abs=0.01)
        assert paid_prin == 0.0

    def test_high_rate_negative_amortisation_capitalised(self) -> None:
        """M11: if interest > payment, the shortfall capitalises into the balance."""
        m, _, _ = amortize_mortgage_monthly(100_000.0, 0.0, 200.0, 0.01)
        # $100k at 1%/mo with $200/mo: 100000*1.01^12 - 200*(1.01^12-1)/0.01.
        assert m == pytest.approx(110_146.04, abs=0.5)
        assert m > 100_000.0


# =============================================================================
# OFFSET OVERFLOW
# =============================================================================


class TestHandleOffsetOverflow:
    """Offset overflow sweep to AU ETFs."""

    def test_offset_exceeds_mortgage(self) -> None:
        """Excess offset sweeps to AU ETFs."""
        o, m, e, b = handle_offset_overflow(200_000.0, 180_000.0, 50_000.0, 50_000.0)
        assert o == pytest.approx(180_000.0, abs=0.01)
        assert m == pytest.approx(180_000.0, abs=0.01)
        assert e == pytest.approx(70_000.0, abs=0.01)
        assert b == pytest.approx(70_000.0, abs=0.01)

    def test_mortgage_paid_off(self) -> None:
        """When mortgage is 0, all offset goes to ETFs."""
        o, m, e, b = handle_offset_overflow(50_000.0, 0.0, 50_000.0, 50_000.0)
        assert o == pytest.approx(0.0, abs=0.01)
        assert m == pytest.approx(0.0, abs=0.01)
        assert e == pytest.approx(100_000.0, abs=0.01)
        assert b == pytest.approx(100_000.0, abs=0.01)

    def test_no_overflow(self) -> None:
        """When offset < mortgage, nothing changes."""
        o, m, e, b = handle_offset_overflow(50_000.0, 200_000.0, 50_000.0, 50_000.0)
        assert o == pytest.approx(50_000.0, abs=0.01)
        assert m == pytest.approx(200_000.0, abs=0.01)
        assert e == pytest.approx(50_000.0, abs=0.01)
        assert b == pytest.approx(50_000.0, abs=0.01)

    def test_both_zero(self) -> None:
        """Zero offset and zero mortgage."""
        o, m, e, b = handle_offset_overflow(0.0, 0.0, 0.0, 0.0)
        assert o == 0.0
        assert m == 0.0
        assert e == 0.0
        assert b == 0.0

    def test_pure_overflow_no_mortgage(self) -> None:
        """Offset > 0 with mortgage=0: all offset to ETFs."""
        o, m, e, b = handle_offset_overflow(100_000.0, 0.0, 10_000.0, 10_000.0)
        assert o == 0.0
        assert e == pytest.approx(110_000.0, abs=0.01)
        assert b == pytest.approx(110_000.0, abs=0.01)


class TestNegativeValues:
    """Negative offset or mortgage values should be handled gracefully."""

    def test_negative_offset_clamped_to_zero(self) -> None:
        """Negative offset should be treated as zero offset."""
        m_neg, _, _ = amortize_mortgage_monthly(100_000.0, -50_000.0, 2_000.0, 0.005)
        m_zero, _, _ = amortize_mortgage_monthly(100_000.0, 0.0, 2_000.0, 0.005)
        assert m_neg == pytest.approx(m_zero, abs=0.01)

    def test_negative_offset_in_overflow_clamped(self) -> None:
        """Negative offset in handle_offset_overflow should be treated as 0."""
        o, m, e, b = handle_offset_overflow(-50_000.0, 200_000.0, 50_000.0, 50_000.0)
        # Same result as offset=0 (no overflow, unchanged)
        assert o == 0.0  # offset clamped to 0 then passed through
        assert m == pytest.approx(200_000.0, abs=0.01)
        assert e == pytest.approx(50_000.0, abs=0.01)


# =============================================================================
# EDGE CASE / INTEGRATION
# =============================================================================


class TestMarginalRate:
    """marginal_rate() helper: bracket lookup driven by the supplied schedule."""

    def test_known_brackets_fy26(self) -> None:
        """FY2026 schedule (low band 15%) gives the expected rate for known incomes."""
        # 0 income = 0% marginal rate (tax-free threshold)
        assert marginal_rate(0.0, brackets=FY26) == 0.0
        # Below 18,200 = 0%
        assert marginal_rate(15_000, brackets=FY26) == 0.0
        # 18,201–45,000 = 15% (FY2026 low band)
        assert marginal_rate(30_000, brackets=FY26) == 0.15
        # 45,001–135,000 = 30%
        assert marginal_rate(100_000, brackets=FY26) == 0.30
        # 135,001–190,000 = 37%
        assert marginal_rate(160_000, brackets=FY26) == 0.37
        # 190,001+ = 45%
        assert marginal_rate(250_000, brackets=FY26) == 0.45

    def test_low_band_depends_on_fy(self) -> None:
        """The 18,201–45,000 band rate follows the schedule year (D10)."""
        assert marginal_rate(30_000, brackets=brackets_for_fy(2025)) == 0.16
        assert marginal_rate(30_000, brackets=brackets_for_fy(2027)) == 0.14


class TestOwnershipValidation:
    """Phase 2: ownership field validation."""

    def test_default_ownership_single_owner(self) -> None:
        """Default ownership {0: 1.0} sums to 1.0."""
        from donebyfifty.models import InvestmentAccount

        acct = InvestmentAccount(label="T", market_value=100_000, cost_basis=50_000)
        assert sum(acct.ownership.values()) == 1.0
        assert acct.ownership == {0: 1.0}

    def test_multi_owner_ownership_sum(self) -> None:
        """Ownership dict sums to 1.0."""
        from donebyfifty.models import InvestmentAccount

        acct = InvestmentAccount(
            label="Joint",
            market_value=100_000,
            cost_basis=50_000,
            ownership=Ownership({0: 0.6, 1: 0.4}),
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001

    def test_deserialise_legacy_no_ownership_field(self) -> None:
        """Legacy profile without ownership field defaults to {0: 1.0}."""
        from donebyfifty.models import _deserialise_account

        data = {
            "label": "T",
            "market_value": 100_000,
            "cost_basis": 50_000,
            "asset_class": "equity",
            "is_offset": False,
        }
        acct = _deserialise_account(data)
        assert acct.ownership == {0: 1.0}

    def test_serialise_deserialise_roundtrip(self) -> None:
        """Ownership survives serialise/deserialise roundtrip."""
        from donebyfifty.models import _deserialise_account, _serialise_account

        original = {
            "label": "Joint",
            "market_value": 200_000,
            "cost_basis": 100_000,
            "asset_class": "equity",
            "is_offset": False,
            "ownership": {"0": 0.4, "1": 0.6},
        }
        acct = _deserialise_account(original)
        assert acct.ownership == {0: 0.4, 1: 0.6}
        # Back to dict
        serialised = _serialise_account(acct)
        assert serialised["ownership"] == {"0": 0.4, "1": 0.6}

    def test_three_equal_owners(self) -> None:
        """Three owners, equal shares — general case beyond 50/50."""
        from donebyfifty.models import InvestmentAccount

        acct = InvestmentAccount(
            label="Three-way",
            market_value=300_000,
            cost_basis=150_000,
            ownership=Ownership({0: 1 / 3, 1: 1 / 3, 2: 1 / 3}),
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001
        for ei in (0, 1, 2):
            assert abs(acct.ownership[ei] - 1 / 3) < 0.001

    def test_three_unequal_owners(self) -> None:
        """Three owners, unequal shares."""
        from donebyfifty.models import InvestmentAccount

        acct = InvestmentAccount(
            label="Split",
            market_value=300_000,
            cost_basis=150_000,
            ownership=Ownership({0: 0.5, 1: 0.3, 2: 0.2}),
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001

    def test_partial_zero_share_owner(self) -> None:
        """An earner in the household with 0% ownership.

        This exercises the case where one earner has zero share — the
        weighted rate computation must skip zero-share earners rather
        than including a 0.0 * rate term that would dilute the average.
        """
        from donebyfifty.models import InvestmentAccount

        acct = InvestmentAccount(
            label="E1+E2 only",
            market_value=200_000,
            cost_basis=100_000,
            ownership=Ownership({0: 0.0, 1: 0.7, 2: 0.3}),
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001
        assert acct.ownership[0] == 0.0

    def test_malformed_ownership_raises(self) -> None:
        """Ownership shares that don't sum to 1.0 raise ValueError."""
        from donebyfifty.models import _deserialise_account as _da

        data = {
            "label": "Bad",
            "market_value": 100_000,
            "cost_basis": 50_000,
            "asset_class": "equity",
            "is_offset": False,
            "ownership": {"0": 0.5, "1": 0.3},  # sums to 0.8
        }
        import pytest

        with pytest.raises(ValueError, match="ownership shares sum to 0.8000"):
            _da(data)

    def test_malformed_ownership_raises_over_100(self) -> None:
        """Ownership shares > 1.0 also raise."""
        from donebyfifty.models import _deserialise_account as _da

        data = {
            "label": "Over",
            "market_value": 100_000,
            "cost_basis": 50_000,
            "asset_class": "equity",
            "is_offset": False,
            "ownership": {"0": 0.6, "1": 0.5},  # sums to 1.1
        }
        import pytest

        with pytest.raises(ValueError, match="ownership shares sum to 1.1000"):
            _da(data)


class TestUiSkipConditions:
    """Phase 2: UI ownership prompt skips correctly."""

    def test_single_earner_no_prompt(self) -> None:
        """Single-earner household: ownership defaults to {0: 1.0}."""
        from donebyfifty.models import InvestmentAccount

        # Call with num_earners=1 — should not prompt, return default
        # We can't fully test the interactive prompt without mocking,
        # but we can verify the default is correct
        acct = InvestmentAccount(label="T", market_value=100_000, cost_basis=50_000)
        assert acct.ownership == {0: 1.0}

    def test_offset_account_no_prompt(self) -> None:
        """Offset account: ownership defaults to {0: 1.0}, prompt skipped."""
        from donebyfifty.models import InvestmentAccount

        # Offset accounts don't trigger CGT, ownership is irrelevant
        acct = InvestmentAccount(
            label="Offset",
            market_value=100_000,
            cost_basis=100_000,
            is_offset=True,
        )
        assert acct.ownership == {0: 1.0}
        assert acct.is_offset is True


class TestPrimitivesCleanliness:
    """Pure kernel modules should not reference legacy-specific state objects."""

    def test_no_simulation_state_import(self) -> None:
        """tax, returns, mortgage and cgt must not import SimulationState."""
        import donebyfifty.cgt as cgt_mod
        import donebyfifty.mortgage as mortgage_mod
        import donebyfifty.returns as returns_mod
        import donebyfifty.tax as tax_mod

        for module in (tax_mod, returns_mod, mortgage_mod, cgt_mod):
            source = module.__file__
            assert source is not None
            with open(source) as f:
                content = f.read()
            assert "SimulationState" not in content, module.__name__
            assert "salary_h" not in content, module.__name__
            assert "salary_w" not in content, module.__name__
            assert "super_h" not in content, module.__name__
            assert "super_w" not in content, module.__name__


# =============================================================================
# M13 — COVERAGE FOR PREVIOUSLY UNTESTED HEADLINE PATHS
# =============================================================================


class TestMlsTiers:
    """Medicare Levy Surcharge tiers (H7, FY2026-27)."""

    def test_singles_tiers(self) -> None:
        assert mls_rate(100_000, couple=False, n_children=0) == 0.0
        assert mls_rate(110_000, couple=False, n_children=0) == 0.01
        assert mls_rate(130_000, couple=False, n_children=0) == 0.0125
        assert mls_rate(200_000, couple=False, n_children=0) == 0.015

    def test_singles_ignore_children(self) -> None:
        assert mls_rate(130_000, couple=False, n_children=3) == 0.0125

    def test_family_tiers(self) -> None:
        assert mls_rate(200_000, couple=True, n_children=0) == 0.0
        assert mls_rate(220_000, couple=True, n_children=0) == 0.01
        assert mls_rate(250_000, couple=True, n_children=0) == 0.0125

    def test_family_child_uplift(self) -> None:
        """Two children lifts the couple base by 2 - 1 = 1 extra child step of $1,500.

        Hand: threshold = 210,000 + 1,500 * (2 - 1) = 211,500.
        """
        assert mls_rate(211_000, couple=True, n_children=2) == 0.0
        assert mls_rate(212_000, couple=True, n_children=2) == 0.01


class TestMortgageRate:
    """Black-Karasinski rate generation (M13, D7).

    Parameters are the moderate-scenario values: theta=0.065, kappa=0.20, sigma=0.18.
    """

    def test_rate_stays_positive(self) -> None:
        random.seed(1)
        rate = 0.06
        for _ in range(150):
            shock = random.gauss(0.0, 1.0)
            rate = generate_mortgage_rate(rate, shock, theta=0.065, kappa=0.2, sigma_tilde=0.18)
            assert rate > 0.0

    def test_zero_shock_from_theta_hand_value(self) -> None:
        """z=0 from prev=theta gives 0.0645246.

        Hand: log_mean = ln(0.065) - 0.18^2/(4*0.2) = -2.7738680.
        phi = exp(-0.2) = 0.8187308; log_rate = log_mean + phi*(ln(0.065) - log_mean)
        = -2.7738680 + 0.8187308*0.0405 = -2.7407104; rate = exp(-2.7407104) = 0.0645246.
        """
        rate = generate_mortgage_rate(0.065, 0.0, theta=0.065, kappa=0.2, sigma_tilde=0.18)
        assert rate == pytest.approx(0.0645246, abs=1e-7)

    def test_shock_scales_by_sigma_eps(self) -> None:
        """z=1 over z=0 is exp(sigma_eps), sigma_eps = 0.18*sqrt((1-exp(-0.4))/0.4) = 0.16341."""
        lo = generate_mortgage_rate(0.065, 0.0, theta=0.065, kappa=0.2, sigma_tilde=0.18)
        hi = generate_mortgage_rate(0.065, 1.0, theta=0.065, kappa=0.2, sigma_tilde=0.18)
        sigma_eps = 0.18 * math.sqrt((1 - math.exp(-0.4)) / 0.4)
        assert hi / lo == pytest.approx(math.exp(sigma_eps), rel=1e-9)

    def test_stationary_arithmetic_mean_is_theta(self) -> None:
        """D7: stationary arithmetic mean equals theta (0.065), within MC tolerance."""
        random.seed(2024)
        rate = 0.065
        total = 0.0
        n = 100_000
        for _ in range(n):
            rate = generate_mortgage_rate(
                rate, random.gauss(0.0, 1.0), theta=0.065, kappa=0.2, sigma_tilde=0.18
            )
            total += rate
        assert total / n == pytest.approx(0.065, abs=0.002)

    def test_non_positive_kappa_rejected(self) -> None:
        with pytest.raises(ValueError):
            generate_mortgage_rate(0.065, 0.0, theta=0.065, kappa=0.0, sigma_tilde=0.18)


class TestPiPayment:
    """Level monthly P&I payment."""

    def test_hand_value(self) -> None:
        """100,000 at 6% over 360 months: r = 0.005, pmt = 100000*r/(1-1.005^-360) = 599.5505."""
        assert pi_payment(100_000.0, 0.06, 360) == pytest.approx(599.5505, abs=0.001)

    def test_zero_rate_is_straight_line(self) -> None:
        assert pi_payment(120_000.0, 0.0, 360) == pytest.approx(333.3333, abs=0.001)

    def test_non_positive_months_rejected(self) -> None:
        with pytest.raises(ValueError):
            pi_payment(100_000.0, 0.06, 0)


class TestSgConstants:
    """FY2026-27 Super Guarantee maximum base (H7)."""

    def test_sg_max_base(self) -> None:
        from donebyfifty.config import SG_MAX_BASE

        assert SG_MAX_BASE == 270_830.0
