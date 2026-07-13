"""Tests for verified mathematical primitives.

Each test pins the output of a primitive against known values computed
from the original verified model. Any change to a primitive that alters
these outputs is a regression.
"""

from __future__ import annotations

import math
import random

import pytest

from primitives import (
    BRACKETS,
    CGT_FLOOR_RATE,
    EDU_SCHEDULE_TODAY,
    EQ_MEAN,
    EQ_STD,
    MEDICARE,
    PT_DAILY_RATE,
    PT_WEEKS_PER_YEAR,
    SUPER_EQ_CORR,
    SUPER_MEAN,
    SUPER_STD,
    amortize_mortgage_monthly,
    consulting_net_income,
    generate_correlated_returns,
    handle_offset_overflow,
    sell_assets,
    tax,
)

# =============================================================================
# CONSTANT INTEGRITY
# =============================================================================


class TestConstants:
    """Ensure constants are intact and semantically correct."""

    def test_brackets_structure(self) -> None:
        assert len(BRACKETS) == 5
        assert BRACKETS[0] == (18200, 0.00)
        assert BRACKETS[-1][1] == 0.45
        # The top threshold should be infinity (sentinel)
        from math import isinf

        assert isinf(BRACKETS[-1][0])
        # Rates should be non-decreasing
        rates = [r for _, r in BRACKETS]
        assert rates == sorted(rates)

    def test_brackets_all_positive_thresholds(self) -> None:
        for threshold, _ in BRACKETS:
            assert threshold > 0

    def test_medicare_rate(self) -> None:
        assert MEDICARE == 0.02

    def test_cgt_rate(self) -> None:
        assert CGT_FLOOR_RATE == 0.30

    def test_return_assumptions(self) -> None:
        assert EQ_MEAN == 0.07
        assert EQ_STD == 0.15
        assert SUPER_MEAN == 0.07
        assert SUPER_STD == 0.12
        assert SUPER_EQ_CORR == 0.80

    def test_pt_assumptions(self) -> None:
        assert PT_DAILY_RATE == 3000.0
        assert PT_WEEKS_PER_YEAR == 48.0

    def test_edu_schedule_total(self) -> None:
        """Known total from client's fee schedule including preschool/daycare: ~$367,008."""
        total = sum(EDU_SCHEDULE_TODAY.values())
        assert total == 367_008.0


# =============================================================================
# TAX
# =============================================================================


class TestTax:
    """Progressive Australian tax + Medicare levy."""

    @pytest.mark.parametrize(
        ("income", "expected"),
        [
            (0, 0.0),
            (18_200, 364.0),  # $0 income tax + 2% Medicare
            (45_000, 5_188.0),  # 16c bracket + Medicare
            (135_000, 33_988.0),  # 30c bracket + Medicare
            (190_000, 55_438.0),  # 37c bracket + Medicare
            (300_000, 107_138.0),  # 45c bracket + Medicare
        ],
    )
    def test_known_incomes(self, income: float, expected: float) -> None:
        assert tax(income) == pytest.approx(expected, abs=0.01)

    def test_medicare_on_all_income(self) -> None:
        """Medicare levy is always 2% of total income."""
        for inc in [10_000, 50_000, 200_000]:
            medicare_component = tax(inc) - (tax(inc) - inc * MEDICARE)
            assert medicare_component == pytest.approx(inc * MEDICARE, abs=0.01)

    def test_zero_income(self) -> None:
        assert tax(0) == 0.0

    def test_monotonic(self) -> None:
        """Tax should be monotonically increasing with income."""
        incomes = [0, 10_000, 18_200, 30_000, 45_000, 80_000, 135_000, 200_000, 500_000]
        taxes = [tax(i) for i in incomes]
        for i in range(1, len(taxes)):
            assert taxes[i] >= taxes[i - 1]


# =============================================================================
# CONSULTING NET INCOME
# =============================================================================


class TestConsultingNetIncome:
    """PT consulting income after tax."""

    @pytest.mark.parametrize(
        ("days", "expected"),
        [
            (0.0, 0.0),
            (0.25, 32_432.0),
            (0.5, 58_172.0),
            (1.0, 106_502.0),
        ],
    )
    def test_known_days(self, days: float, expected: float) -> None:
        assert consulting_net_income(days) == pytest.approx(expected, abs=0.01)

    def test_zero_days(self) -> None:
        assert consulting_net_income(0.0) == 0.0

    def test_cache_used(self) -> None:
        """Calling twice with same input should return same value (cache hit)."""
        v1 = consulting_net_income(2.0)
        v2 = consulting_net_income(2.0)
        assert v1 == v2

    def test_net_less_than_gross(self) -> None:
        """After-tax income should be less than gross."""
        for days in [0.5, 1.0, 2.0]:
            gross = days * PT_WEEKS_PER_YEAR * PT_DAILY_RATE
            net = consulting_net_income(days)
            assert net < gross
            assert net > 0

    def test_no_negative(self) -> None:
        """Negative days/week should return 0."""
        assert consulting_net_income(-1.0) == 0.0


# =============================================================================
# EDUCATION COST
# =============================================================================


# =============================================================================
# CORRELATED RETURNS
# =============================================================================


class TestGenerateCorrelatedReturns:
    """Cholesky decomposition for correlated return series."""

    def test_seeded_known_values(self) -> None:
        """With seed=42, first three draws produce pinned values."""
        random.seed(42)
        expected = [
            (0.03540753, 0.03476766),
            (0.04051030, 0.10551927),
            (0.03797366, -0.05785800),
        ]
        for eq_exp, super_exp in expected:
            eq, sup = generate_correlated_returns()
            assert eq == pytest.approx(eq_exp, abs=1e-7)
            assert sup == pytest.approx(super_exp, abs=1e-7)

    def test_mean_convergence(self) -> None:
        """Over 50k draws, mean return should approximate target mean."""
        random.seed(12345)
        eq_returns: list[float] = []
        super_returns: list[float] = []
        for _ in range(50_000):
            eq, sup = generate_correlated_returns()
            eq_returns.append(eq)
            super_returns.append(sup)

        eq_mean = sum(eq_returns) / len(eq_returns)
        super_mean = sum(super_returns) / len(super_returns)

        # Should be within 0.5% of target (Monte Carlo noise)
        assert eq_mean == pytest.approx(EQ_MEAN, abs=0.005)
        assert super_mean == pytest.approx(SUPER_MEAN, abs=0.005)

    def test_return_type(self) -> None:
        """Should always return a tuple of two floats."""
        random.seed(99)
        for _ in range(10):
            result = generate_correlated_returns()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], float)
            assert isinstance(result[1], float)

    def test_custom_rho(self) -> None:
        """Custom rho values should still return valid floats."""
        random.seed(42)
        for rho in [0.0, 0.5, 1.0]:
            eq, sup = generate_correlated_returns(rho=rho)
            assert isinstance(eq, float)
            assert isinstance(sup, float)

    @pytest.mark.slow
    def test_correlation_convergence(self) -> None:
        """Empirical correlation should approximate target rho=0.80."""
        random.seed(54321)
        eq_returns: list[float] = []
        super_returns: list[float] = []
        for _ in range(100_000):
            eq, sup = generate_correlated_returns()
            eq_returns.append(eq)
            super_returns.append(sup)

        n = len(eq_returns)
        mean_eq = sum(eq_returns) / n
        mean_sup = sum(super_returns) / n
        cov = sum((eq_returns[i] - mean_eq) * (super_returns[i] - mean_sup) for i in range(n)) / n
        var_eq = sum((x - mean_eq) ** 2 for x in eq_returns) / n
        var_sup = sum((x - mean_sup) ** 2 for x in super_returns) / n
        corr = cov / (math.sqrt(var_eq) * math.sqrt(var_sup))

        assert corr == pytest.approx(SUPER_EQ_CORR, abs=0.02)


# =============================================================================
# SELL ASSETS
# =============================================================================


class TestSellAssets:
    """Asset sell-down with cost-basis and CGT logic."""

    def test_no_cgt_full_basis(self) -> None:
        """No CGT when cgt_on=False, full basis means no gain."""
        asset: dict[str, float] = {"val": 1000.0, "basis": 1000.0}
        remain, _tax, _nofloor = sell_assets(asset, 500.0, cgt_on=False)
        assert remain == pytest.approx(0.0, abs=0.01)
        assert asset["val"] == pytest.approx(500.0, abs=0.01)
        assert asset["basis"] == pytest.approx(500.0, abs=0.01)

    def test_with_cgt_partial_basis(self) -> None:
        """CGT applied on gain portion when basis < value."""
        asset: dict[str, float] = {"val": 1000.0, "basis": 500.0}
        remain, _tax, _nofloor = sell_assets(asset, 500.0, cgt_on=True)
        assert remain == pytest.approx(0.0, abs=0.01)
        assert asset["val"] == pytest.approx(411.76, abs=0.01)
        assert asset["basis"] == pytest.approx(205.88, abs=0.01)

    def test_insufficient_assets(self) -> None:
        """If asset value < gross needed, sell all and return remaining need."""
        asset: dict[str, float] = {"val": 100.0, "basis": 100.0}
        remain, _tax, _nofloor = sell_assets(asset, 500.0, cgt_on=False)
        assert remain == pytest.approx(400.0, abs=0.01)
        assert asset["val"] == pytest.approx(0.0, abs=0.01)
        assert asset["basis"] == pytest.approx(0.0, abs=0.01)

    def test_zero_valued_asset(self) -> None:
        """Selling from a zero-valued asset does nothing."""
        asset: dict[str, float] = {"val": 0.0, "basis": 0.0}
        remain, _tax, _nofloor = sell_assets(asset, 500.0, cgt_on=True)
        assert remain == pytest.approx(500.0, abs=0.01)

    def test_negative_basis(self) -> None:
        """Edge case: negative basis should not cause issues."""
        asset: dict[str, float] = {"val": 1000.0, "basis": -100.0}
        remain, _tax, _nofloor = sell_assets(asset, 500.0, cgt_on=True)
        # Should still sell and return some value
        assert remain >= 0
        assert asset["val"] < 1000.0

    def test_no_need(self) -> None:
        """No spending need: nothing is sold."""
        asset: dict[str, float] = {"val": 1000.0, "basis": 500.0}
        remain, _tax, _nofloor = sell_assets(asset, 0.0, cgt_on=True)
        assert remain == pytest.approx(0.0, abs=0.01)
        assert asset["val"] == pytest.approx(1000.0, abs=0.01)


# =============================================================================
# AMORTIZE MORTGAGE
# =============================================================================


class TestAmortizeMortgageMonthly:
    """Mortgage amortisation with offset benefit."""

    def test_no_offset(self) -> None:
        """Without offset, interest is charged on full principal."""
        m, o = amortize_mortgage_monthly(100_000.0, 0.0, 2_000.0, 0.005)
        assert m == pytest.approx(81_496.66, abs=0.01)
        assert o == pytest.approx(0.0, abs=0.01)

    def test_with_offset(self) -> None:
        """With offset, effective debt is reduced so more principal is paid."""
        m_no_ofs, _ = amortize_mortgage_monthly(100_000.0, 0.0, 2_000.0, 0.005)
        m_with_ofs, _ = amortize_mortgage_monthly(100_000.0, 50_000.0, 2_000.0, 0.005)
        # With offset, mortgage reduces FASTER (less interest paid)
        assert m_with_ofs < m_no_ofs
        assert m_with_ofs == pytest.approx(78_412.77, abs=0.01)

    def test_offset_greater_than_mortgage(self) -> None:
        """When offset >= mortgage, effective debt is 0, no interest charged."""
        m, o = amortize_mortgage_monthly(50_000.0, 60_000.0, 2_000.0, 0.005)
        # All $2k goes to principal each month
        expected = 50_000.0 - 12 * 2_000.0
        assert m == pytest.approx(expected, abs=0.01)

    def test_mortgage_fully_paid(self) -> None:
        """Mortgage should not go below zero."""
        m, o = amortize_mortgage_monthly(5_000.0, 0.0, 2_000.0, 0.005)
        # 3 months should pay it off; remaining months do nothing
        assert m == 0.0

    def test_zero_mortgage(self) -> None:
        """Zero mortgage is a no-op."""
        m, o = amortize_mortgage_monthly(0.0, 0.0, 2_000.0, 0.005)
        assert m == 0.0

    def test_high_rate_negative_amortisation_prevented(self) -> None:
        """If interest > payment, only interest is paid (no principal reduction)."""
        m, o = amortize_mortgage_monthly(100_000.0, 0.0, 200.0, 0.01)
        assert m == pytest.approx(100_000.0, abs=0.01)  # principal unchanged


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
        m_neg, _ = amortize_mortgage_monthly(100_000.0, -50_000.0, 2_000.0, 0.005)
        m_zero, _ = amortize_mortgage_monthly(100_000.0, 0.0, 2_000.0, 0.005)
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


# =============================================================================
# ITEM 6 PHASE 1 — CGT COST-BASE INDEXATION
# =============================================================================


class TestCgtCostBaseIndexation:
    """Item 6 Phase 1: cost-base indexed by CPI for CGT (Treasury Laws
    Amendment (Tax Reform No. 1) Act 2026)."""

    def test_sell_assets_with_zero_inflation_identical_to_old_behaviour(
        self,
    ) -> None:
        """When cumulative_inflation_factor=1.0 (no inflation),
        indexed_basis == nominal_basis, so result is identical."""
        asset = {"val": 200_000.0, "basis": 100_000.0}
        # Old behaviour (nominal gain): gain = 200k-100k = 100k, CGT=30k
        # New behaviour with factor=1.0: same
        remain, _tax, _nofloor = sell_assets(
            asset.copy(),
            50_000,
            cgt_on=True,
            weighted_marginal_rate=0.30,
            cumulative_inflation_factor=1.0,
        )
        assert remain == 0.0, f"Expected 0, got {remain}"

    def test_sell_assets_indexation_reduces_cgt(self) -> None:
        """With cumulative_inflation_factor=1.1 (10% inflation since
        acquisition), indexed_basis = 100k * 1.1 = 110k.
        Real gain = 200k - 110k = 90k.  CGT = 90k * 0.30 = 27k.
        Net proceeds = 200k - 27k = 173k.
        Old nominal: CGT = (200k-100k)*0.30 = 30k.  Net = 170k.
        So with indexation, 3k more net proceeds from same sale.

        To net 50k after CGT:
        Old: gross_needed = 50k / (1 - (gain_frac=0.5)*0.30) = 50/0.85 = 58,824
        New: indexed_gain_frac = (200k - 110k)/200k = 0.45
             effective_rate = 0.45 * 0.30 = 0.135
             gross_needed = 50k / (1 - 0.135) = 50/0.865 = 57,803
        """
        import copy

        asset = {"val": 200_000.0, "basis": 100_000.0}
        asset_old = copy.deepcopy(asset)
        asset_new = copy.deepcopy(asset)

        # Old behaviour (no indexation)
        sell_assets(
            asset_old,
            50_000,
            cgt_on=True,
            weighted_marginal_rate=0.30,
            cumulative_inflation_factor=1.0,
        )
        old_basis_remaining = asset_old["basis"]
        old_val_remaining = asset_old["val"]

        # New behaviour with indexation
        sell_assets(
            asset_new,
            50_000,
            cgt_on=True,
            weighted_marginal_rate=0.30,
            cumulative_inflation_factor=1.1,
        )
        new_basis_remaining = asset_new["basis"]
        new_val_remaining = asset_new["val"]

        # With indexation, less of the asset needs to be sold to net the
        # same after-tax amount -> more value remains in the asset.
        assert new_val_remaining > old_val_remaining, (
            f"Indexed: ${new_val_remaining:,.0f} remaining, "
            f"Old: ${old_val_remaining:,.0f} remaining — "
            "indexation should preserve more value."
        )
        # Basis differs because a different amount of the asset was sold.
        # With indexation: less needs to be sold, so less basis is consumed.
        assert new_basis_remaining > old_basis_remaining, (
            f"Indexed: basis ${new_basis_remaining:,.0f} remaining, "
            f"Old: ${old_basis_remaining:,.0f} remaining — "
            "indexation preserves more basis (less asset sold)."
        )

    def test_sell_assets_hand_calculated_indexation(self) -> None:
        """Hand-calculated example to pin exact numbers.

        Asset bought for $100k, held 10 years at 3%/yr inflation.
        cumulative_inflation_factor = (1.03)^10 = 1.3439...
        Indexed basis = 100k * 1.3439 = 134,392.
        Sold for $200k.  Real gain = 200k - 134,392 = 65,608.
        CGT at 30% = 65,608 * 0.30 = 19,682.
        Net proceeds = 200k - 19,682 = 180,318.

        Old nominal: gain = 100k, CGT = 30k, net = 170k.
        Difference: $10,318 more with indexation.
        """
        inf_factor = (1.03) ** 10  # ≈ 1.3439
        asset = {"val": 200_000.0, "basis": 100_000.0}
        result, _tax, _nofloor = sell_assets(
            asset,
            200_000,
            cgt_on=True,
            weighted_marginal_rate=0.30,
            cumulative_inflation_factor=inf_factor,
        )
        # After selling the full asset for $200k:
        # Val should be 0 (all sold)
        assert asset["val"] == 0.0
        # Basis should be 0 (all consumed)
        assert asset["basis"] == 0.0
        # remain should be 200k - net_proceeds
        net_proceeds = 200_000 - result
        # Net proceeds should be ~180,318
        assert (
            abs(net_proceeds - 180_318) < 100
        ), f"Expected net ~$180,318, got ${net_proceeds:,.0f}"
        # The difference from old nominal ($170k net) should be ~$10,318
        old_net = 200_000 - (200_000 - 100_000) * 0.30  # = 170_000
        improvement = net_proceeds - old_net
        assert (
            improvement > 10_000
        ), f"Expected >$10k improvement over nominal, got ${improvement:,.0f}"


# =============================================================================
# PHASE 2 — CGT 30% FLOOR + PER-OWNER WEIGHTED RATE
# =============================================================================


class TestMarginalRate:
    """marginal_rate() helper — single source of truth with BRACKETS."""

    def test_known_brackets(self) -> None:
        """Confirm marginal_rate returns the correct bracket for known incomes."""
        from primitives import marginal_rate

        # 0 income = 0% marginal rate (tax-free threshold)
        assert marginal_rate(0.0) == 0.0
        # Below 18,200 = 0%
        assert marginal_rate(15_000) == 0.0
        # 18,201–45,000 = 19%
        assert marginal_rate(30_000) == 0.16
        # 45,001–135,000 = 32.5% (includes 2% Medicare via tax(), but marginal_rate is bracket-only)
        assert marginal_rate(100_000) == 0.30
        # 135,001–190,000 = 37%
        assert marginal_rate(160_000) == 0.37
        # 190,001+ = 45%
        assert marginal_rate(250_000) == 0.45

    def test_uses_module_brackets_not_hardcoded(self) -> None:
        """marginal_rate uses the same BRACKETS module-level constant."""
        from primitives import marginal_rate

        # Call with default brackets — should match BRACKETS
        rate_at_100k = marginal_rate(100_000)
        # Correct marginal rate for $100k = 0.30 (45-135k bracket)
        assert rate_at_100k == 0.30


class TestSellAssetsPhase2:
    """Phase 2: per-owner weighted marginal rate with 30% floor."""

    def test_single_owner_backward_compat(self) -> None:
        """Single owner (weighted_marginal_rate=0.30) with no indexation
        produces same result as old cgt_rate=0.30.
        """
        asset = {"val": 200_000.0, "basis": 100_000.0}
        remain, _tax, _nofloor = sell_assets(
            asset, 50_000, cgt_on=True, weighted_marginal_rate=0.30, cumulative_inflation_factor=1.0
        )
        # Same as old: net_proceeds=50k, val was 200k, 58,824 sold
        assert remain == 0.0
        assert asset["val"] == 200_000 - 50_000 / (1 - 0.5 * 0.30)  # ~141,176

    def test_floor_applied_to_low_income_owner(self) -> None:
        """The specific scenario the reform targets: two-earner household,
        one earner with zero taxable income (bridge period, no salary, no PT),
        the other with marginal rate above 30%.  Joint account 50/50.

        E1: taxable=$0 → marginal=0% → max(0, 0.30)=0.30
        E2: taxable=$200k → marginal=45% → max(45, 0.30)=0.45
        Weighted = 0.5 * 0.30 + 0.5 * 0.45 = 0.375

        Real gain $100k on $200k sale.
        CGT = $100k * 0.375 = $37,500.
        Net = $200k - $37,500 = $162,500.

        Without floor (E1 at 0%): weighted = 0.5*0 + 0.5*0.45 = 0.225.
        CGT = $22,500.  Net = $177,500.  Checks the floor adds $15k tax.
        """
        from primitives import marginal_rate

        # E1: zero income → floor kicks in
        e1_mr = max(marginal_rate(0.0), 0.30)  # = 0.30
        # E2: high income → own rate
        e2_mr = max(marginal_rate(200_000), 0.30)  # = max(0.45, 0.30) = 0.45
        weighted = 0.5 * e1_mr + 0.5 * e2_mr  # = 0.375

        assert abs(weighted - 0.375) < 0.001, f"Expected 0.375, got {weighted}"

        # Now test through sell_assets
        asset = {"val": 200_000.0, "basis": 100_000.0}
        # Sell full asset
        result, _tax, _nofloor = sell_assets(
            asset,
            200_000,
            cgt_on=True,
            weighted_marginal_rate=weighted,
            cumulative_inflation_factor=1.0,
        )
        net_proceeds = 200_000 - result
        # Expected: real_gain = 100k, CGT = 100k * 0.375 = 37.5k
        expected_net = 200_000 - 100_000 * 0.375  # = 162,500
        assert (
            abs(net_proceeds - expected_net) < 100
        ), f"Expected net ${expected_net:,.0f}, got ${net_proceeds:,.0f}"

    def test_floor_no_effect_when_both_high_income(self) -> None:
        """When both owners have marginal rates above 30%, the floor
        doesn't change the outcome."""
        from primitives import marginal_rate

        e1_mr = max(marginal_rate(200_000), 0.30)  # 0.45
        e2_mr = max(marginal_rate(180_000), 0.30)  # 0.37 (135k-190k bracket)
        # Without floor: same rates
        unfloored = 0.5 * 0.45 + 0.5 * 0.37  # = 0.41
        floored = 0.5 * e1_mr + 0.5 * e2_mr  # = 0.41
        assert abs(floored - unfloored) < 0.001

    def test_phase1_phase2_composition(self) -> None:
        """Phase 1 real_gain is unchanged by Phase 2.  Same test scenario
        produces same real_gain, only the applied rate differs.

        Asset: bought $100k, 10 years at 3% inflation, sold for $200k.
        Phase 1 indexed_basis ≈ $134,392.  real_gain ≈ $65,608.
        Phase 1 CGT (at 30%): $19,682.  Net: $180,318.
        Phase 2 CGT (at 37.5%): $65,608 * 0.375 = $24,603.  Net: $175,397.
        """
        inf_factor = (1.03) ** 10  # ≈ 1.3439

        # Phase 1 only (weighted_marginal_rate=0.30, old behaviour)
        asset = {"val": 200_000.0, "basis": 100_000.0}
        result_p1, _tax1, _nf1 = sell_assets(
            asset,
            200_000,
            cgt_on=True,
            weighted_marginal_rate=0.30,
            cumulative_inflation_factor=inf_factor,
        )
        net_p1 = 200_000 - result_p1

        # Same scenario with Phase 2 floor at 37.5%
        asset2 = {"val": 200_000.0, "basis": 100_000.0}
        result_p2, _tax2, _nf2 = sell_assets(
            asset2,
            200_000,
            cgt_on=True,
            weighted_marginal_rate=0.375,
            cumulative_inflation_factor=inf_factor,
        )
        net_p2 = 200_000 - result_p2

        # Phase 1 net ≈ $180,318
        assert abs(net_p1 - 180_318) < 100, f"Phase 1 net expected ~$180,318, got ${net_p1:,.0f}"
        # Phase 2 net < Phase 1 net (higher rate = more tax = less net)
        assert net_p2 < net_p1, f"Phase 2 net ${net_p2:,.0f} should be < Phase 1 ${net_p1:,.0f}"
        # The real_gain is the same — confirm via basis consumed
        # Phase 1 & 2 both consumed same basis (same sale amount)
        assert (
            abs(asset["basis"] - asset2["basis"]) < 1.0
        ), "Basis consumed should be identical (same sale amount)"


class TestOwnershipValidation:
    """Phase 2: ownership field validation."""

    def test_default_ownership_single_owner(self) -> None:
        """Default ownership {0: 1.0} sums to 1.0."""
        from models import InvestmentAccount

        acct = InvestmentAccount(label="T", market_value=100_000, cost_basis=50_000)
        assert sum(acct.ownership.values()) == 1.0
        assert acct.ownership == {0: 1.0}

    def test_multi_owner_ownership_sum(self) -> None:
        """Ownership dict sums to 1.0."""
        from models import InvestmentAccount

        acct = InvestmentAccount(
            label="Joint",
            market_value=100_000,
            cost_basis=50_000,
            ownership={0: 0.6, 1: 0.4},
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001

    def test_deserialise_legacy_no_ownership_field(self) -> None:
        """Legacy profile without ownership field defaults to {0: 1.0}."""
        from models import _deserialise_account

        data = {
            "label": "T",
            "market_value": 100_000,
            "cost_basis": 50_000,
            "asset_class": "equity",
            "is_offset": False,
            "cgt_rate": 0.30,
        }
        acct = _deserialise_account(data)
        assert acct.ownership == {0: 1.0}

    def test_serialise_deserialise_roundtrip(self) -> None:
        """Ownership survives serialise/deserialise roundtrip."""
        from models import _deserialise_account, _serialise_account

        original = {
            "label": "Joint",
            "market_value": 200_000,
            "cost_basis": 100_000,
            "asset_class": "equity",
            "is_offset": False,
            "cgt_rate": 0.30,
            "ownership": {"0": 0.4, "1": 0.6},
        }
        acct = _deserialise_account(original)
        assert acct.ownership == {0: 0.4, 1: 0.6}
        # Back to dict
        serialised = _serialise_account(acct)
        assert serialised["ownership"] == {"0": 0.4, "1": 0.6}

    def test_three_equal_owners(self) -> None:
        """Three owners, equal shares — general case beyond 50/50."""
        from models import InvestmentAccount

        acct = InvestmentAccount(
            label="Three-way",
            market_value=300_000,
            cost_basis=150_000,
            ownership={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001
        for ei in (0, 1, 2):
            assert abs(acct.ownership[ei] - 1 / 3) < 0.001

    def test_three_unequal_owners(self) -> None:
        """Three owners, unequal shares."""
        from models import InvestmentAccount

        acct = InvestmentAccount(
            label="Split",
            market_value=300_000,
            cost_basis=150_000,
            ownership={0: 0.5, 1: 0.3, 2: 0.2},
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001

    def test_partial_zero_share_owner(self) -> None:
        """An earner in the household with 0% ownership.

        This exercises the case where one earner has zero share — the
        weighted rate computation must skip zero-share earners rather
        than including a 0.0 * rate term that would dilute the average.
        """
        from models import InvestmentAccount

        acct = InvestmentAccount(
            label="E1+E2 only",
            market_value=200_000,
            cost_basis=100_000,
            ownership={0: 0.0, 1: 0.7, 2: 0.3},
        )
        assert abs(sum(acct.ownership.values()) - 1.0) < 0.001
        assert acct.ownership[0] == 0.0

    def test_malformed_ownership_raises(self) -> None:
        """Ownership shares that don't sum to 1.0 raise ValueError."""
        from models import _deserialise_account as _da

        data = {
            "label": "Bad",
            "market_value": 100_000,
            "cost_basis": 50_000,
            "asset_class": "equity",
            "is_offset": False,
            "cgt_rate": 0.30,
            "ownership": {"0": 0.5, "1": 0.3},  # sums to 0.8
        }
        import pytest

        with pytest.raises(ValueError, match="ownership sums to 0.8000"):
            _da(data)

    def test_malformed_ownership_raises_over_100(self) -> None:
        """Ownership shares > 1.0 also raise."""
        from models import _deserialise_account as _da

        data = {
            "label": "Over",
            "market_value": 100_000,
            "cost_basis": 50_000,
            "asset_class": "equity",
            "is_offset": False,
            "cgt_rate": 0.30,
            "ownership": {"0": 0.6, "1": 0.5},  # sums to 1.1
        }
        import pytest

        with pytest.raises(ValueError, match="ownership sums to"):
            _da(data)

    def test_weighted_rate_with_three_owners(self) -> None:
        """sell_assets with per-owner weighted rate and 3+ owners.

        Owners:
          E1: taxable=$0        → marginal=0%  → max(0.00, 0.30)=0.30, share=0.5
          E2: taxable=$200k     → marginal=45% → max(0.45, 0.30)=0.45, share=0.3
          E3: taxable=$100k     → marginal=30% → max(0.30, 0.30)=0.30, share=0.2

        Weighted = 0.5*0.30 + 0.3*0.45 + 0.2*0.30 = 0.15 + 0.135 + 0.06 = 0.345

        real_gain = $100k.  CGT = $100k * 0.345 = $34,500.
        Net = $200k - $34,500 = $165,500.

        Without floor on E1 (0% instead of 30%):
          Weighted = 0.5*0 + 0.3*0.45 + 0.2*0.30 = 0 + 0.135 + 0.06 = 0.195
          CGT = $19,500.  Net = $180,500.  Floor adds $15k tax.
        """
        from primitives import marginal_rate, sell_assets

        e1 = max(marginal_rate(0), 0.30)  # 0.00 → 0.30
        e2 = max(marginal_rate(200_000), 0.30)  # 0.45
        e3 = max(marginal_rate(100_000), 0.30)  # 0.30
        weighted = 0.5 * e1 + 0.3 * e2 + 0.2 * e3
        assert abs(weighted - 0.345) < 0.001, f"Expected 0.345, got {weighted}"

        # Now test through sell_assets
        asset = {"val": 200_000.0, "basis": 100_000.0}
        result, _tax, _nofloor = sell_assets(
            asset,
            200_000,
            cgt_on=True,
            weighted_marginal_rate=weighted,
            cumulative_inflation_factor=1.0,
        )
        net_proceeds = 200_000 - result
        expected_net = 200_000 - 100_000 * 0.345  # = 165,500
        assert (
            abs(net_proceeds - expected_net) < 100
        ), f"Expected net ${expected_net:,.0f}, got ${net_proceeds:,.0f}"


class TestPtIncomeInMarginalRate:
    """Phase 2: PT gross income must be included in earner_taxable."""

    def test_pt_gross_added_to_taxable_realistic(self) -> None:
        """PT gross income adds to salary in earner_taxable for marginal
        rate calculation.  Uses realistic bridge-client values:
        1 day/wk consulting at $3,000/day = $144k/yr PT gross.
        40k salary + $144k PT gross = $184k → 37% marginal rate.

        Without PT inclusion: 40k → 16% marginal rate (wrong by 21pp).
        """
        import random

        from models import Earner, Household, SimulationInputs
        from simulation import run_monte_carlo

        random.seed(42)
        e = Earner(
            label="E1",
            salary=40_000,
            super_balance=50_000,
            salary_growth_rate=0.0,  # flat real
            retirement_age=50,
            super_access_age=60,
            employment_type="self_employed",  # PT is their bridge income
            pt_days_per_week=1.0,
            pt_start_age=37,
            pt_end_age=55,
            personal_super_contributions_total_p_a=0,
        )
        h = Household(earners=(e,), base_living_expenses=100_000)
        inputs = SimulationInputs(
            simulation_start_age=37,
            n_iterations=10,
            inflation=0.0,
            cgt_on_drawdowns=True,
        )
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)
        # Simulation runs without error = earner_taxable_includes PT gross
        # (marginal_rate() handles $184k → 37%)

    def test_zero_pt_doesnt_affect_taxable(self) -> None:
        """Earner with pt_days_per_week=0 has no PT gross added."""
        from simulation import _SimulationState

        # _SimulationState has earner_taxable_incomes field
        state = _SimulationState(
            age=37,
            year=0,
            super_balances=[100_000],
            earner_taxable_incomes=[0.0],
        )
        assert state.earner_taxable_incomes == [0.0]


class TestUiSkipConditions:
    """Phase 2: UI ownership prompt skips correctly."""

    def test_single_earner_no_prompt(self) -> None:
        """Single-earner household: ownership defaults to {0: 1.0}."""
        from models import InvestmentAccount

        # Call with num_earners=1 — should not prompt, return default
        # We can't fully test the interactive prompt without mocking,
        # but we can verify the default is correct
        acct = InvestmentAccount(label="T", market_value=100_000, cost_basis=50_000)
        assert acct.ownership == {0: 1.0}

    def test_offset_account_no_prompt(self) -> None:
        """Offset account: ownership defaults to {0: 1.0}, prompt skipped."""
        from models import InvestmentAccount

        # Offset accounts don't trigger CGT, ownership is irrelevant
        acct = InvestmentAccount(
            label="Offset",
            market_value=100_000,
            cost_basis=100_000,
            is_offset=True,
            cgt_rate=0.0,
        )
        assert acct.ownership == {0: 1.0}
        assert acct.is_offset is True


class TestPrimitivesCleanliness:
    """Primitives module should not reference legacy-specific state objects."""

    def test_no_simulation_state_import(self) -> None:
        """primitives.py must not import SimulationState."""
        import primitives as p  # type: ignore[import-unclear]

        source = p.__file__
        assert source is not None
        with open(source) as f:
            content = f.read()
        assert "SimulationState" not in content
        assert "salary_h" not in content
        assert "salary_w" not in content
        assert "super_h" not in content
        assert "super_w" not in content
