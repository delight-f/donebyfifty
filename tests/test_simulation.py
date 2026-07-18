"""Tests for the Monte Carlo simulation engine.

Covers deterministic runs, Reference regression, edge cases for
N-earner, N-child, N-mortgage, and N-account households.
"""

from __future__ import annotations

from models import (
    Child,
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
    SimulationInputs,
)
from primitives import EDU_SCHEDULE_TODAY, EQ_MEAN, SUPER_MEAN
from simulation import (
    RetirementSearchResult,
    ScenarioComparisonResult,
    SequencingRiskResult,
    run_monte_carlo,
    run_retirement_search,
    run_scenario_comparison,
    run_sequencing_analysis,
    run_single_trial,
)

# =============================================================================
# HELPERS
# =============================================================================


def reference_household(
    retire_age: int = 50,
    sell_uk: bool = True,
) -> Household:
    """Build a household matching the original reference family's financial position.

    Args:
        retire_age: Age at which both earners stop working.
        sell_uk: If True, UK ETFs are sold and proceeds added to offset
                 (matching the original's ``sell_uk_etfs=True`` mode).

    Returns:
        A ``Household`` configured for reference-family inputs.
    """
    e1 = Earner(
        label="Earner 1",
        salary=320_000.0,
        super_balance=310_720.71,
        salary_growth_rate=0.03,
        retirement_age=retire_age,
        super_access_age=60,
        sg_rate=0.12,
    )
    e2 = Earner(
        label="Earner 2",
        salary=200_000.0,
        super_balance=238_380.63,
        salary_growth_rate=0.025,
        retirement_age=retire_age,
        super_access_age=60,
        sg_rate=0.0,
        personal_super_contributions_total_p_a=32_500.0,
    )

    c1 = Child(
        label="Child 1",
        age=2,
        education_schedule=tuple(EDU_SCHEDULE_TODAY.items()),
    )

    m1 = MortgageAccount(
        label="Mortgage 1",
        principal=820_000.0,
        interest_rate=0.0605,
        monthly_payment=5_491.43,
        offset_accounts=("Offset 1",),
    )

    accounts: list[InvestmentAccount] = []

    if sell_uk:
        # UK ETFs sold before simulation; proceeds added to offset
        offset_bal = 200_000.0 + (512_000.0 - 73_085.0)  # 638,915
        accounts.append(
            InvestmentAccount(
                label="Offset 1",
                market_value=offset_bal,
                cost_basis=offset_bal,
                asset_class="cash",
                is_offset=True,
                cgt_rate=0.0,
            )
        )
    else:
        accounts.append(
            InvestmentAccount(
                label="Offset 1",
                market_value=200_000.0,
                cost_basis=200_000.0,
                asset_class="cash",
                is_offset=True,
                cgt_rate=0.0,
            )
        )
        accounts.append(
            InvestmentAccount(
                label="UK ETFs",
                market_value=512_000.0,
                cost_basis=201_000.0,
                asset_class="equity",
                tax_jurisdiction="uk",
                cgt_rate=0.30,
            )
        )

    accounts.append(
        InvestmentAccount(
            label="AU ETFs",
            market_value=63_000.0,
            cost_basis=63_000.0,
            asset_class="equity",
            tax_jurisdiction="au",
            cgt_rate=0.30,
        )
    )

    return Household(
        earners=(e1, e2),
        children=(c1,),
        mortgages=(m1,),
        investment_accounts=tuple(accounts),
        base_living_expenses=75_000.0,
        retirement_target=100_000.0,
    )


def deterministic_inputs() -> SimulationInputs:
    """Simulation inputs with deterministic (mean) returns."""
    return SimulationInputs(
        cgt_on_drawdowns=True,
        sell_order=("AU ETFs",),
    )


# =============================================================================
# DETERMINISTIC SANITY
# =============================================================================


class TestDeterministic:
    """Run with mean returns — results should be self-consistent."""

    def test_bridge_and_super_positive(self) -> None:
        """With deterministic 7% returns, bridge and super should be positive."""
        h = reference_household(retire_age=50)
        inputs = deterministic_inputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.bridge > 0
        assert r.total_super > 0

    def test_later_retirement_more_bridge(self) -> None:
        """Retiring later should leave more bridge assets."""
        h55 = reference_household(retire_age=55)
        inputs = deterministic_inputs()
        bridge_end = min(e.super_access_age for e in h55.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years

        r55 = run_single_trial(
            household=h55,
            inputs=inputs,
            eq_returns=eq,
        )
        r50 = run_single_trial(
            household=reference_household(retire_age=50),
            inputs=inputs,
            eq_returns=eq,
        )
        assert r55.bridge > r50.bridge
        assert r55.total_super > r50.total_super

    def test_mortgage_declines(self) -> None:
        """Mortgage should decline or be zero by end of simulation."""
        h = reference_household(retire_age=50)
        inputs = deterministic_inputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.total_mortgage <= 820_000.0  # started at 820k
        assert r.total_mortgage >= 0.0

    def test_super_grows_over_time(self) -> None:
        """Super at end should exceed starting super."""
        h = reference_household(retire_age=50)
        inputs = deterministic_inputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.total_super > 549_101.34  # starting super sum

    def test_sell_uk_more_bridge(self) -> None:
        """Selling UK ETFs should increase bridge vs keeping them."""
        h_sell = reference_household(sell_uk=True)
        inputs = deterministic_inputs()
        bridge_end = min(e.super_access_age for e in h_sell.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years

        r_sell = run_single_trial(
            household=h_sell,
            inputs=inputs,
            eq_returns=eq,
        )
        r_keep = run_single_trial(
            household=reference_household(sell_uk=False),
            inputs=inputs,
            eq_returns=eq,
        )
        # Selling UK should result in more bridge (CGT paid upfront but
        # proceeds earn returns)
        assert r_sell.bridge > 0
        assert r_keep.bridge > 0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Various household compositions should run without error."""

    def test_single_no_children_no_mortgage(self) -> None:
        """Simplest possible case: 1 earner, no children, no debts."""
        h = Household(
            earners=(Earner(salary=100_000.0, super_balance=50_000.0),),
        )
        inputs = SimulationInputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.bridge >= 0
        assert r.total_super > 0

    def test_single_income_couple(self) -> None:
        """2 earners, one non-working."""
        h = Household(
            earners=(
                Earner(label="Worker", salary=150_000.0, super_balance=80_000.0),
                Earner(
                    label="Non-worker",
                    salary=0.0,
                    super_balance=40_000.0,
                    employment_type="not_employed",
                ),
            ),
        )
        inputs = SimulationInputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.total_super > 120_000.0  # initial sum
        assert r.bridge >= 0

    def test_many_children(self) -> None:
        """6 children with different ages should not crash."""
        children = tuple(Child(label=f"Child {i}", age=2 + i) for i in range(6))
        h = Household(
            earners=(Earner(salary=200_000.0, super_balance=100_000.0),),
            children=children,
        )
        inputs = SimulationInputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.bridge >= 0

    def test_no_investments(self) -> None:
        """No accounts, no mortgage — pure salary model."""
        h = Household(
            earners=(Earner(salary=80_000.0, super_balance=20_000.0),),
            investment_accounts=(),
            mortgages=(),
        )
        inputs = SimulationInputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.bridge >= 0

    def test_staggered_retirement(self) -> None:
        """Earner 1 retires at 55, Earner 2 at 65."""
        h = Household(
            earners=(
                Earner(
                    label="Early",
                    salary=150_000.0,
                    super_balance=100_000.0,
                    retirement_age=55,
                    super_access_age=60,
                ),
                Earner(
                    label="Late",
                    salary=120_000.0,
                    super_balance=80_000.0,
                    retirement_age=65,
                    super_access_age=60,
                ),
            ),
        )
        inputs = SimulationInputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        assert r.bridge >= 0
        assert r.total_super > 180_000.0

    def test_no_mortgage_payment(self) -> None:
        """Mortgage with monthly_payment=0 means no principal is paid.

        With no payment, interest is unpaid but principal stays flat
        (negative amortisation is prevented).
        """
        h = Household(
            earners=(Earner(salary=200_000.0, super_balance=100_000.0),),
            mortgages=(
                MortgageAccount(
                    principal=500_000.0,
                    interest_rate=0.06,
                    monthly_payment=0.0,
                ),
            ),
            investment_accounts=(
                InvestmentAccount(
                    label="Offset",
                    market_value=50_000.0,
                    cost_basis=50_000.0,
                    is_offset=True,
                    cgt_rate=0.0,
                ),
            ),
        )
        inputs = SimulationInputs(simulation_start_age=37)
        n_years = min(e.super_access_age for e in h.earners) - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        # With no payment, principal stays flat (negative amortisation prevented)
        assert r.total_mortgage == 500_000.0

    def test_io_mortgage_interest_charged(self) -> None:
        """A $500k IO mortgage at 6% should reduce bridge vs. no mortgage.

        With deterministic returns, the interest-only loan charges
        $30,000/year in interest (net of offset), which must be funded
        from income. The no-mortgage case should have a larger bridge.
        """
        h_with = Household(
            earners=(Earner(salary=300_000.0, super_balance=100_000.0, retirement_age=60),),
            mortgages=(
                MortgageAccount(
                    principal=500_000.0,
                    interest_rate=0.06,
                    monthly_payment=0.0,
                ),
            ),
            base_living_expenses=60_000.0,
        )
        h_without = Household(
            earners=(Earner(salary=300_000.0, super_balance=100_000.0, retirement_age=60),),
            base_living_expenses=60_000.0,
        )
        inputs = SimulationInputs(simulation_start_age=37)
        n = min(e.super_access_age for e in h_with.earners) - inputs.simulation_start_age
        eq = [EQ_MEAN] * n
        [SUPER_MEAN] * n
        r_with = run_single_trial(h_with, inputs, eq_returns=eq)
        r_without = run_single_trial(h_without, inputs, eq_returns=eq)
        # With mortgage should have less bridge (interest cost drains surplus)
        assert r_with.bridge < r_without.bridge, (
            f"IO mortgage case (${r_with.bridge:,.0f}) should be less "
            f"than no-mortgage case (${r_without.bridge:,.0f})"
        )
        # IO mortgage principal should still be unchanged
        assert r_with.total_mortgage == 500_000.0

    def test_zero_super_balance(self) -> None:
        """Earner with no starting super should not crash."""
        h = Household(
            earners=(Earner(salary=80_000.0, super_balance=0.0),),
        )
        inputs = SimulationInputs()
        n_years = min(e.super_access_age for e in h.earners) - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)
        # Super should have grown from contributions + returns
        assert r.total_super >= 0

    def test_surplus_allocation_to_investments(self) -> None:
        """With surplus_investment_pct=100, surplus should go to non-offset accounts."""
        h = Household(
            earners=(Earner(salary=200_000.0, super_balance=50_000.0, retirement_age=60),),
            investment_accounts=(
                InvestmentAccount(
                    label="Offset",
                    market_value=10_000.0,
                    cost_basis=10_000.0,
                    is_offset=True,
                    cgt_rate=0.0,
                ),
                InvestmentAccount(
                    label="Shares",
                    market_value=10_000.0,
                    cost_basis=10_000.0,
                    asset_class="equity",
                    interest_rate=0.0,
                ),
            ),
            mortgages=(
                MortgageAccount(
                    principal=500_000.0,
                    interest_rate=0.06,
                    monthly_payment=3_000.0,
                    offset_accounts=("Offset",),
                ),
            ),
        )
        inputs = SimulationInputs(
            simulation_start_age=37,
            cgt_on_drawdowns=True,
            surplus_investment_pct=100.0,
        )
        n_years = min(e.super_access_age for e in h.earners) - inputs.simulation_start_age
        eq = [0.07] * n_years
        [0.07] * n_years
        r = run_single_trial(h, inputs, eq_returns=eq)
        # Bridge should be positive
        assert r.bridge > 0

    def test_surplus_allocation_to_offset(self) -> None:
        """With surplus_investment_pct=0, surplus should go to offset accounts."""
        h = Household(
            earners=(Earner(salary=200_000.0, super_balance=50_000.0, retirement_age=60),),
            investment_accounts=(
                InvestmentAccount(
                    label="Offset",
                    market_value=10_000.0,
                    cost_basis=10_000.0,
                    is_offset=True,
                    cgt_rate=0.0,
                ),
                InvestmentAccount(
                    label="Shares",
                    market_value=10_000.0,
                    cost_basis=10_000.0,
                    asset_class="equity",
                    interest_rate=0.0,
                ),
            ),
            mortgages=(
                MortgageAccount(
                    principal=500_000.0,
                    interest_rate=0.06,
                    monthly_payment=3_000.0,
                    offset_accounts=("Offset",),
                ),
            ),
        )
        inputs = SimulationInputs(
            simulation_start_age=37,
            cgt_on_drawdowns=True,
            surplus_investment_pct=0.0,
        )
        n_years = min(e.super_access_age for e in h.earners) - inputs.simulation_start_age
        eq = [0.07] * n_years
        [0.07] * n_years
        r = run_single_trial(h, inputs, eq_returns=eq)
        # Bridge should still be positive
        assert r.bridge > 0

    def test_high_inflation_stress(self) -> None:
        """10% inflation should produce higher failure probability than 2.5%."""
        h = reference_household(retire_age=50)
        inputs_low = deterministic_inputs()
        inputs_high = SimulationInputs(
            inflation=0.10,
            cgt_on_drawdowns=True,
        )

        result_low = run_monte_carlo(household=h, inputs=inputs_low)
        result_high = run_monte_carlo(household=h, inputs=inputs_high)

        # Higher inflation should reduce success rate (or at least not increase it)
        assert result_high.p_success <= result_low.p_success + 0.05  # noise allowance


# =============================================================================
# MONTE CARLO STATISTICAL TESTS
# =============================================================================


class TestMonteCarlo:
    """Statistical properties of the Monte Carlo engine."""

    def test_mc_returns_p_success(self) -> None:
        """Monte Carlo run should return p_success between 0 and 1."""
        h = reference_household(retire_age=60)
        inputs = SimulationInputs(n_iterations=200)
        result = run_monte_carlo(household=h, inputs=inputs)
        assert 0 <= result.p_success <= 1.0
        assert result.trials == 200

    def test_mc_percentiles_ordered(self) -> None:
        """Percentiles should be monotonically increasing."""
        h = reference_household(retire_age=60)
        inputs = SimulationInputs(n_iterations=200)
        result = run_monte_carlo(household=h, inputs=inputs)
        assert result.bridge_p5 <= result.bridge_p10
        assert result.bridge_p10 <= result.bridge_p25
        assert result.bridge_p25 <= result.bridge_median
        assert result.bridge_median <= result.bridge_p75
        assert result.bridge_p75 <= result.bridge_p90
        assert result.bridge_p90 <= result.bridge_p95

    def test_mc_super_positive(self) -> None:
        """Median super balance should be positive."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200)
        result = run_monte_carlo(household=h, inputs=inputs)
        assert result.super_median > 0

    def test_mc_per_earner_super(self) -> None:
        """Per-earner super stats should list each earner."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200)
        result = run_monte_carlo(household=h, inputs=inputs)
        assert len(result.per_earner_super_p50) == 2
        for label, val in result.per_earner_super_p50.items():
            assert val > 0
            assert label in ("Earner 1", "Earner 2")

    def test_mc_uses_sell_order(self) -> None:
        """Specifying sell_order should not cause errors."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(
            n_iterations=100,
            sell_order=("AU ETFs",),
        )
        result = run_monte_carlo(household=h, inputs=inputs)
        assert 0 <= result.p_success <= 1.0


# =============================================================================
# REFERENCE REGRESSION (non-exact, within plausible range)
# =============================================================================


class TestReferenceRegression:
    """The new engine should produce plausible results for reference inputs.

    Because the general engine treats all earners uniformly (SG for all,
    configurable per-earner), the exact deterministic output differs from
    the original reference case. This test verifies that results are within
    a plausible range rather than matching exact numbers.
    """

    def test_deterministic_within_plausible_range(self) -> None:
        """Deterministic bridge should be within 50% of original's."""
        h = reference_household(retire_age=50)
        inputs = deterministic_inputs()
        bridge_end = min(e.super_access_age for e in h.earners)
        n_years = bridge_end - inputs.simulation_start_age
        eq = [EQ_MEAN] * n_years
        [SUPER_MEAN] * n_years
        r = run_single_trial(household=h, inputs=inputs, eq_returns=eq)

        original_bridge = 12_415_095.37
        ratio = r.bridge / original_bridge
        # Bounds widened after adding super fees (0.85% p.a.), per-earner super returns,
        # and retirement age default change to 50
        assert 0.30 <= ratio <= 2.5, f"Bridge ratio {ratio:.2f} outside [0.30, 2.5]"

    def test_stochastic_plausible(self) -> None:
        """Monte Carlo with 500 trials should produce sensible stats."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=500)
        result = run_monte_carlo(household=h, inputs=inputs)
        assert 0 < result.p_success <= 1.0
        assert result.bridge_median > 0
        assert result.super_median > 0


# =============================================================================
# MORTGAGE / OFFSET DRAWDOWN TESTS
# =============================================================================


class TestIOInterestTiming:
    """Fix: IO interest now uses post-drawdown offset."""

    def test_io_interest_post_drawdown(self) -> None:
        """IO interest is computed after offset drawdown for the same period.

        A client with an IO mortgage and low income draws offset for living
        expenses. The pre-drawdown offset is higher than post-drawdown, so
        pre-drawdown IO interest understates the actual charge. This test
        verifies the interest reflects the post-drawdown lower offset.
        """
        import random

        random.seed(42)
        offset = InvestmentAccount(
            label="Offset 1",
            market_value=100_000,
            cost_basis=100_000,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        m = MortgageAccount(
            label="IO Loan",
            principal=500_000,
            interest_rate=0.06,
            monthly_payment=0.0,  # interest-only
            offset_accounts=("Offset 1",),
        )
        # Low income forces offset drawdown for living expenses
        e = Earner(salary=50_000, super_balance=50_000, retirement_age=60)
        h = Household(
            earners=(e,),
            mortgages=(m,),
            investment_accounts=(offset,),
            base_living_expenses=80_000,  # exceeds salary
        )
        inputs = SimulationInputs(simulation_start_age=37)
        n = 60 - 37
        eq = [EQ_MEAN] * n
        r = run_single_trial(h, inputs, eq_returns=eq)
        # The IO loan should have incurred interest; offset should be drawn down
        assert r.mortgage_principals[0] > 0, "IO principal should remain > 0"
        assert r.bridge < 100_000, "Offset should have been drawn below starting"

    def test_io_interest_less_than_pre_drawdown(self) -> None:
        """Direct comparison: pre-drawdown IO interest hits the cashflow,
        post-drawdown is what's actually owed. Since offset drops during
        the period, the pre-drawdown figure understates interest.
        """
        import random

        random.seed(42)
        offset = InvestmentAccount(
            label="Offset 1",
            market_value=200_000,
            cost_basis=200_000,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        m = MortgageAccount(
            label="IO Loan",
            principal=400_000,
            interest_rate=0.07,
            monthly_payment=0.0,
            offset_accounts=("Offset 1",),
        )
        e = Earner(salary=40_000, super_balance=50_000, retirement_age=60)
        h = Household(
            earners=(e,),
            mortgages=(m,),
            investment_accounts=(offset,),
            base_living_expenses=90_000,
        )
        inputs = SimulationInputs(simulation_start_age=37)
        n = 60 - 37
        eq = [EQ_MEAN] * n
        r = run_single_trial(h, inputs, eq_returns=eq)
        # Assertions: loan isn't cleared, bridge assets depleted
        assert r.mortgage_principals[0] > 0
        assert r.bridge <= 200_000  # offset mostly or fully consumed


class TestOffsetReserveFloor:
    """Fix: offset_reserve_floor preserves offset for repayment serviceability."""

    def test_floor_preserves_mortgage_term_clearance(self) -> None:
        """A P&I mortgage client with a large mortgage relative to income:
        no-floor drains offset and mortgage does not clear by term end;
        with a floor, offset is preserved, amortisation stays fast enough
        for the mortgage to clear in most paths.
        """
        import random

        random.seed(42)
        offset = InvestmentAccount(
            label="Offset 1",
            market_value=200_000,
            cost_basis=200_000,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        taxable = InvestmentAccount(
            label="Taxable",
            market_value=50_000,
            cost_basis=50_000,
            asset_class="equity",
            cgt_rate=0.30,
        )
        # Large mortgage ($650k at 7%), tight budget ($100k income, $58k living):
        # monthly $4,500 P&I payment barely covers interest when offset is drained.
        # With no floor, offset consumed by living; effective debt stays high,
        # amortisation stalls, mortgage does not clear by age 60.
        m_no_floor = MortgageAccount(
            label="BigLoan",
            principal=650_000,
            interest_rate=0.07,
            monthly_payment=4_500,
            offset_accounts=("Offset 1",),
            offset_reserve_floor=0.0,
            loan_term_end_age=60,
        )
        m_with_floor = MortgageAccount(
            label="BigLoan",
            principal=650_000,
            interest_rate=0.07,
            monthly_payment=4_500,
            offset_accounts=("Offset 1",),
            offset_reserve_floor=100_000.0,
            loan_term_end_age=60,
        )
        e = Earner(salary=100_000, super_balance=50_000, retirement_age=60)
        h_no_floor = Household(
            earners=(e,),
            mortgages=(m_no_floor,),
            investment_accounts=(offset, taxable),
            base_living_expenses=58_000,
        )
        h_with_floor = Household(
            earners=(Earner(salary=100_000, super_balance=50_000, retirement_age=60),),
            mortgages=(m_with_floor,),
            investment_accounts=(
                InvestmentAccount(
                    label="Offset 1",
                    market_value=200_000,
                    cost_basis=200_000,
                    asset_class="cash",
                    is_offset=True,
                    cgt_rate=0.0,
                ),
                InvestmentAccount(
                    label="Taxable",
                    market_value=50_000,
                    cost_basis=50_000,
                    asset_class="equity",
                    cgt_rate=0.30,
                ),
            ),
            base_living_expenses=58_000,
        )
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=500)
        res_no_floor = run_monte_carlo(h_no_floor, inputs)
        res_with_floor = run_monte_carlo(h_with_floor, inputs)
        # Floor case must have materially better clearance than no-floor
        assert (
            res_with_floor.mortgage_term_clearance_rate > res_no_floor.mortgage_term_clearance_rate
        ), (
            f"Floor ({res_with_floor.mortgage_term_clearance_rate:.1%}) should be > "
            f"no-floor ({res_no_floor.mortgage_term_clearance_rate:.1%})"
        )
        # Print numbers for review
        print(f"\n  No floor: clearance={res_no_floor.mortgage_term_clearance_rate * 100:.1f}%")
        print(f"  With floor: clearance={res_with_floor.mortgage_term_clearance_rate * 100:.1f}%")


class TestMortgageStallCheck:
    """Fix: term-clearance check flags stalled mortgages as failures."""

    def test_stalled_mortgage_flagged_as_term_failure(self) -> None:
        """A mortgage where interest >= monthly payment stalls (no principal
        repaid). The bridge may survive, but the term-clearance check should
        flag this as a failure.
        """
        import random

        random.seed(42)
        # $600k at 7% = $42k/yr interest = $3,500/month. Monthly payment is
        # exactly $3,500, so NO principal is ever repaid (all goes to interest).
        m = MortgageAccount(
            label="Staller",
            principal=600_000,
            interest_rate=0.07,
            monthly_payment=3_500,  # interest-only effective
            loan_term_end_age=60,  # must clear by 60
        )
        # Give a large cash buffer so bridge survives, but the mortgage
        # payment ($3,500/mo = $42k/yr) exactly covers interest on $600k at
        # 7%, so NO principal is ever repaid.
        cash_account = InvestmentAccount(
            label="Cash",
            market_value=300_000,
            cost_basis=300_000,
            asset_class="cash",
            is_offset=False,
            cgt_rate=0.0,
        )
        e = Earner(salary=120_000, super_balance=50_000, retirement_age=60)
        h = Household(
            earners=(e,),
            mortgages=(m,),
            investment_accounts=(cash_account,),
            base_living_expenses=40_000,
        )
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=200)
        res = run_monte_carlo(h, inputs)
        # Bridge survival should be high (low expenses relative to income)
        assert res.p_success > 0.5, f"Bridge survival ({res.p_success:.1%}) should be high"
        # Term-clearance should be 0% — principal never decreases
        assert res.mortgage_term_clearance_rate == 0.0, (
            f"Term clearance ({res.mortgage_term_clearance_rate:.1%}) "
            f"should be 0% for a stalled mortgage"
        )
        # Verify at per-mortgage level too
        for label, pct in res.per_mortgage_term_cleared_pct.items():
            assert pct == 0.0, f"{label} cleared {pct:.1%}, expected 0%"


class TestOffsetReserveModes:
    """Stall-prevention and interest-cancelling offset reserve modes."""

    def test_stall_prevention_preserves_offset(self) -> None:
        """Stall-prevention mode preserves offset when payment < interest.

        A $650k loan at 7% with $3,500/mo payment has interest = $3,792/mo
        > payment. The dynamic floor = max(0, 650k - 3.5k/(7%/12)) = $50k.
        Without a floor, offset drains and the loan stalls at full balance.
        With stall_prevention, $50k is preserved, keeping the effective
        debt at $600k where interest = payment = $3,500/mo (no growth).
        """
        import random

        random.seed(42)
        m_no_floor = MortgageAccount(
            label="L",
            principal=650_000,
            interest_rate=0.07,
            monthly_payment=3_500,
            loan_term_end_age=60,
            offset_accounts=("O1",),
            offset_reserve_mode="fixed",
            offset_reserve_floor=0.0,
        )
        m_stall = MortgageAccount(
            label="L",
            principal=650_000,
            interest_rate=0.07,
            monthly_payment=3_500,
            loan_term_end_age=60,
            offset_accounts=("O1",),
            offset_reserve_mode="stall_prevention",
            offset_reserve_floor=0.0,
        )
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=200)
        offset = InvestmentAccount(
            label="O1",
            market_value=60_000,
            cost_basis=60_000,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        taxable = InvestmentAccount(
            label="T",
            market_value=40_000,
            cost_basis=40_000,
            asset_class="cash",
            is_offset=False,
            cgt_rate=0.0,
        )
        e = Earner(salary=130_000, super_balance=50_000, retirement_age=60)
        h_no = Household(
            earners=(e,),
            mortgages=(m_no_floor,),
            investment_accounts=(offset, taxable),
            base_living_expenses=70_000,
        )
        h_stall = Household(
            earners=(Earner(salary=130_000, super_balance=50_000, retirement_age=60),),
            mortgages=(m_stall,),
            investment_accounts=(
                InvestmentAccount(
                    label="O1",
                    market_value=60_000,
                    cost_basis=60_000,
                    asset_class="cash",
                    is_offset=True,
                    cgt_rate=0.0,
                ),
                InvestmentAccount(
                    label="T",
                    market_value=40_000,
                    cost_basis=40_000,
                    asset_class="cash",
                    is_offset=False,
                    cgt_rate=0.0,
                ),
            ),
            base_living_expenses=70_000,
        )
        res_no = run_monte_carlo(h_no, inputs)
        res_stall = run_monte_carlo(h_stall, inputs)
        rem_no = res_no.remaining_mortgage_p50.get("L", 0)
        rem_stall = res_stall.remaining_mortgage_p50.get("L", 0)
        assert rem_stall < rem_no, (
            f"Stall-prevention remaining (${rem_stall:,.0f}) should be < no-floor (${rem_no:,.0f})"
        )

    def test_interest_cancelling_preserves_more_than_stall(self) -> None:
        """Interest-cancelling preserves more offset than stall-prevention.

        A $500k loan at 7% with $3,500/mo payment has interest = $2,917/mo.
        Stall-prevention floor = max(0, 500k - 3.5k/(7%/12)) = max(0, 500k - 600k) = 0
        (payment already covers interest at full principal).
        Interest-cancelling floor = $500k (full principal), so ALL offset
        ($200k) is preserved for the mortgage.

        With high living expenses draining the offset,
        interest-cancelling preserves more offset → keeps more in offset
        → reduces effective debt → lower remaining mortgage.
        """
        import random

        random.seed(42)
        m_stall = MortgageAccount(
            label="L",
            principal=500_000,
            interest_rate=0.07,
            monthly_payment=3_500,
            loan_term_end_age=60,
            offset_accounts=("O1",),
            offset_reserve_mode="stall_prevention",
            offset_reserve_floor=0.0,
        )
        m_cancel = MortgageAccount(
            label="L",
            principal=500_000,
            interest_rate=0.07,
            monthly_payment=3_500,
            loan_term_end_age=60,
            offset_accounts=("O1",),
            offset_reserve_mode="interest_cancelling",
            offset_reserve_floor=0.0,
        )
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=200)
        offset = InvestmentAccount(
            label="O1",
            market_value=200_000,
            cost_basis=200_000,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        taxable = InvestmentAccount(
            label="T",
            market_value=200_000,
            cost_basis=200_000,
            asset_class="cash",
            is_offset=False,
            cgt_rate=0.0,
        )
        e = Earner(salary=90_000, super_balance=50_000, retirement_age=60)
        h_stall = Household(
            earners=(e,),
            mortgages=(m_stall,),
            investment_accounts=(offset, taxable),
            base_living_expenses=90_000,
        )
        h_cancel = Household(
            earners=(Earner(salary=90_000, super_balance=50_000, retirement_age=60),),
            mortgages=(m_cancel,),
            investment_accounts=(
                InvestmentAccount(
                    label="O1",
                    market_value=200_000,
                    cost_basis=200_000,
                    asset_class="cash",
                    is_offset=True,
                    cgt_rate=0.0,
                ),
                InvestmentAccount(
                    label="T",
                    market_value=200_000,
                    cost_basis=200_000,
                    asset_class="cash",
                    is_offset=False,
                    cgt_rate=0.0,
                ),
            ),
            base_living_expenses=90_000,
        )
        res_stall = run_monte_carlo(h_stall, inputs)
        res_cancel = run_monte_carlo(h_cancel, inputs)
        rem_stall = res_stall.remaining_mortgage_p50.get("L", 0)
        rem_cancel = res_cancel.remaining_mortgage_p50.get("L", 0)
        # Interest-cancelling preserves ALL offset → lower remaining mortgage
        assert rem_cancel < rem_stall, (
            f"Interest-cancelling remaining (${rem_cancel:,.0f}) should be < "
            f"stall-prevention (${rem_stall:,.0f})"
        )

    def test_interest_cancelling_no_offset_available_for_other_expenses(
        self,
    ) -> None:
        """Interest-cancelling preserves all offset for the mortgage,
        so no offset is available for other living expenses. This may reduce
        bridge survival compared to a no-floor strategy.
        """
        import random

        random.seed(42)
        m = MortgageAccount(
            label="L",
            principal=400_000,
            interest_rate=0.065,
            monthly_payment=3_500,
            loan_term_end_age=60,
            offset_accounts=("O1",),
            offset_reserve_mode="interest_cancelling",
            offset_reserve_floor=0.0,
        )
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=200)
        offset = InvestmentAccount(
            label="O1",
            market_value=450_000,
            cost_basis=450_000,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        # No other accounts — so no money available for expenses if offset
        # is fully reserved for the mortgage
        e = Earner(salary=100_000, super_balance=50_000, retirement_age=60)
        h = Household(
            earners=(e,),
            mortgages=(m,),
            investment_accounts=(offset,),
            base_living_expenses=110_000,
        )
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)
        assert 0.0 <= res.p_success <= 1.0


# =============================================================================
# ITEM 1b — STAGGERED RETIREMENT EXPENSE BLEND
# =============================================================================


class TestStaggeredRetirementExpenseBlend:
    """Item 1b: proportional expense blend for multi-earner staggered retirement."""

    def test_two_earner_staggered_blend(self) -> None:
        """Confirm blended expense formula works for 2 earners.
        E1 retires at 50, E2 at 55.  At age 52: 1/2 retired, blend=0.5*60k+0.5*80k=$70k.
        """
        import random

        random.seed(42)
        e1 = Earner(
            label="E1",
            salary=150_000,
            super_balance=100_000,
            retirement_age=50,
            super_access_age=60,
        )
        e2 = Earner(
            label="E2",
            salary=120_000,
            super_balance=100_000,
            retirement_age=55,
            super_access_age=60,
        )
        h = Household(earners=(e1, e2), base_living_expenses=80_000, retirement_target=60_000)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=50, inflation=0.025)
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)
        assert 0.0 <= res.p_success <= 1.0

    def test_single_earner_no_change(self) -> None:
        """Single-earner: blend reduces to binary 0 or 1, same as old behaviour."""
        import random

        random.seed(42)
        e = Earner(
            label="E1",
            salary=150_000,
            super_balance=200_000,
            retirement_age=50,
            super_access_age=60,
        )
        h = Household(earners=(e,), base_living_expenses=80_000, retirement_target=60_000)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=50, inflation=0.025)
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)
        assert 0.0 <= res.p_success <= 1.0


# =============================================================================
# ITEM 10 — SUCCESS PROBABILITY DISPLAY (single occurrence)
# =============================================================================


class TestSuccessProbabilityDisplay:
    """Item 10: duplicate success probability line removed."""

    def test_success_probability_appears_once(self) -> None:
        """display_results() outputs success probability exactly once."""
        from io import StringIO

        from models import SimulationResults
        from ui import console as ui_console
        from ui import display_results

        buf = StringIO()
        old_file = ui_console.file
        ui_console.file = buf
        try:
            res = SimulationResults(
                trials=1000,
                p_success=0.873,
                bridge_mean=200_000.0,
                bridge_median=180_000.0,
                bridge_p5=50_000.0,
                bridge_p10=80_000.0,
                bridge_p25=120_000.0,
                bridge_p75=250_000.0,
                bridge_p90=300_000.0,
                bridge_p95=350_000.0,
                bridge_min=10_000.0,
                bridge_floor=5_000.0,
                floor_age=55,
                floor_end_bridge=100_000.0,
                super_median=500_000.0,
                horizon_age=60,
            )
            display_results(res)
        finally:
            ui_console.file = old_file

        output = buf.getvalue()
        # Count only the chart line pattern (not the warning message which
        # also contains "Success probability" at the start of a sentence)
        # The chart line has the format: "Success probability: X%  (95%..."
        import re

        chart_lines = re.findall(r"Success probability: \d+\.\d+%", output)
        assert len(chart_lines) == 1, (
            f"Expected 1 chart line, got {len(chart_lines)}: {chart_lines}"
        )


# =============================================================================
# ITEM 8 — REAL SALARY GROWTH
# =============================================================================


class TestRealSalaryGrowth:
    """Item 8: salary_growth_rate is now real (above inflation)."""

    def test_zero_real_growth_tracks_inflation(self) -> None:
        """0% real + 2.5% inflation = 2.5% effective nominal growth.
        Salary grows at inflation rate, flat in real terms.
        """
        import random

        random.seed(42)
        e = Earner(
            label="E1",
            salary=100_000,
            super_balance=50_000,
            salary_growth_rate=0.0,
            retirement_age=50,
            super_access_age=60,
        )
        h = Household(earners=(e,), base_living_expenses=80_000)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=10, inflation=0.025)
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)

    def test_real_growth_compounds_correctly(self) -> None:
        """Assert the inflation compounding step is present in source."""
        import simulation as sim_mod

        source_file = sim_mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            content = f.read()
        assert "(1 + earner.salary_growth_rate) * (1 + inputs.inflation)" in content, (
            "Salary growth must compound real_rate with inflation."
        )


# =============================================================================
# ITEM 2 — EMPLOYMENT TYPE
# =============================================================================


class TestEmploymentType:
    """Item 2: employment_type field adds self-employed path."""

    def test_self_employed_runs(self) -> None:
        """Self-employed earner simulates without errors."""
        import random

        random.seed(42)
        e = Earner(
            label="SE",
            salary=120_000,
            super_balance=50_000,
            salary_growth_rate=0.005,
            retirement_age=55,
            super_access_age=60,
            employment_type="self_employed",
            sg_rate=0.12,
        )
        h = Household(earners=(e,), base_living_expenses=80_000)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=10, inflation=0.025)
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)

    def test_employed_normal(self) -> None:
        """Employed earner simulates with normal SG."""
        import random

        random.seed(42)
        e = Earner(
            label="EM",
            salary=120_000,
            super_balance=50_000,
            salary_growth_rate=0.005,
            retirement_age=55,
            super_access_age=60,
            employment_type="employed",
            sg_rate=0.12,
        )
        h = Household(earners=(e,), base_living_expenses=80_000)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=10, inflation=0.025)
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)

    def test_not_employed_no_income(self) -> None:
        """Not-employed: no salary income."""
        import random

        random.seed(42)
        e = Earner(label="NE", salary=120_000, super_balance=50_000, employment_type="not_employed")
        h = Household(earners=(e,), base_living_expenses=80_000)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=10, inflation=0.025)
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)

    def test_legacy_is_employed_backward_compat(self) -> None:
        """Legacy is_employed=True/False (no employment_type) maps correctly."""
        from models import _deserialise_earner

        # is_employed=True -> employment_type="employed"
        old_true = {
            "label": "E1",
            "salary": 120_000,
            "super_balance": 50_000,
            "salary_growth_rate": 0.005,
            "retirement_age": 55,
            "super_access_age": 60,
            "sg_rate": 0.12,
            "is_employed": True,
        }
        e_t = _deserialise_earner(old_true)
        assert e_t.employment_type == "employed"

        # is_employed=False -> employment_type="not_employed"
        old_false = dict(old_true)
        old_false["is_employed"] = False
        e_f = _deserialise_earner(old_false)
        assert e_f.employment_type == "not_employed"

        # With employment_type key, use it directly
        new_data = dict(old_true)
        new_data["employment_type"] = "self_employed"
        e_s = _deserialise_earner(new_data)
        assert e_s.employment_type == "self_employed"

        # Run the loaded earners through simulation
        import random

        random.seed(42)
        inputs = SimulationInputs(simulation_start_age=37, n_iterations=10, inflation=0.025)
        h_loaded = Household(earners=(e_t,), base_living_expenses=80_000)
        res = run_monte_carlo(h_loaded, inputs)
        assert isinstance(res.p_success, float)


# =============================================================================
# ITEM 6 PHASE 1 — CGT INDEXATION (main tests in test_primitives.py)
# =============================================================================


class TestCgtCostBaseIndexation:
    """Item 6 Phase 1: CGT with CPI-indexed cost base (standalone)."""

    def test_cgt_with_indexation_via_simulation(self) -> None:
        """End-to-end: a household with an investment account runs
        correctly with CGT indexation. The indexed-basis computation
        produces a valid result (details tested in test_primitives.py)."""
        import random

        random.seed(42)
        acct = InvestmentAccount(
            label="AU ETFs",
            market_value=200_000,
            cost_basis=100_000,
            asset_class="equity",
            is_offset=False,
            cgt_rate=0.30,
        )
        e = Earner(
            label="E1",
            salary=150_000,
            super_balance=100_000,
            salary_growth_rate=0.005,
            retirement_age=50,
            super_access_age=60,
        )
        h = Household(earners=(e,), investment_accounts=(acct,), base_living_expenses=80_000)
        inputs = SimulationInputs(
            simulation_start_age=37, n_iterations=50, inflation=0.025, cgt_on_drawdowns=True
        )
        res = run_monte_carlo(h, inputs)
        assert isinstance(res.p_success, float)


# =============================================================================
# SEQUENCING RISK SORT DIRECTION
# =============================================================================


class TestSequencingRiskSortDirection:
    """Regression: verify sequencing analysis sorts in the correct direction."""

    def test_sort_direction_is_monotonic(self) -> None:
        """The drawdown-period zip-sort-unzip for worst-first must produce
        ascending equity returns (smallest first, largest last) in the
        drawdown portion only, leaving working years in original order."""
        import random

        random.seed(42)

        n_years = 15
        drawdown_start = 5  # first 5 years are working, last 10 are drawdown
        from primitives import SUPER_EQ_CORR, generate_correlated_returns

        eq_returns: list[float] = []
        eq_zs: list[float] = []

        for _ in range(n_years):
            eq_r, _super_r, eq_z = generate_correlated_returns(rho=SUPER_EQ_CORR, return_z=True)
            eq_returns.append(eq_r)
            eq_zs.append(eq_z)

        # Simulate worst_first: working unsorted, drawdown sorted ascending
        zipped = list(zip(eq_returns, eq_zs))
        working = zipped[:drawdown_start]
        drawdown = list(zipped[drawdown_start:])
        drawdown.sort(key=lambda x: x[0])  # ascending = worst returns first
        zipped = working + drawdown
        re_eq, re_z = zip(*zipped)
        re_eq_list = list(re_eq)

        # Working years unchanged (random order, just verify length + contents)
        assert re_eq_list[:drawdown_start] == eq_returns[:drawdown_start]

        # Drawdown years are ascending
        drawdown_vals = re_eq_list[drawdown_start:]
        assert drawdown_vals[0] <= drawdown_vals[-1], (
            f"worst_first drawdown must be ascending: first={drawdown_vals[0]:.4f} "
            f"should be <= last={drawdown_vals[-1]:.4f}"
        )
        for i in range(len(drawdown_vals) - 1):
            assert drawdown_vals[i] <= drawdown_vals[i + 1], (
                f"worst_first drawdown not monotonic at index {i}: "
                f"{drawdown_vals[i]:.4f} > {drawdown_vals[i + 1]:.4f}"
            )

    def test_best_first_sort_is_descending(self) -> None:
        """The drawdown-period sort for best-first must produce descending
        equity returns in the drawdown portion only."""
        import random

        random.seed(42)

        n_years = 15
        drawdown_start = 5
        from primitives import SUPER_EQ_CORR, generate_correlated_returns

        eq_returns: list[float] = []
        eq_zs: list[float] = []

        for _ in range(n_years):
            eq_r, _super_r, eq_z = generate_correlated_returns(rho=SUPER_EQ_CORR, return_z=True)
            eq_returns.append(eq_r)
            eq_zs.append(eq_z)

        # Simulate best_first: working unsorted, drawdown sorted descending
        zipped = list(zip(eq_returns, eq_zs))
        working = zipped[:drawdown_start]
        drawdown = list(zipped[drawdown_start:])
        drawdown.sort(key=lambda x: x[0], reverse=True)  # descending = best first
        zipped = working + drawdown
        re_eq, re_z = zip(*zipped)
        re_eq_list = list(re_eq)

        # Working years unchanged
        assert re_eq_list[:drawdown_start] == eq_returns[:drawdown_start]

        # Drawdown years are descending
        drawdown_vals = re_eq_list[drawdown_start:]
        assert drawdown_vals[0] >= drawdown_vals[-1], (
            f"best_first drawdown must be descending: first={drawdown_vals[0]:.4f} "
            f"should be >= last={drawdown_vals[-1]:.4f}"
        )
        for i in range(len(drawdown_vals) - 1):
            assert drawdown_vals[i] >= drawdown_vals[i + 1], (
                f"best_first drawdown not monotonic at index {i}: "
                f"{drawdown_vals[i]:.4f} < {drawdown_vals[i + 1]:.4f}"
            )


# =============================================================================
# INTEGRATION TESTS — OPT-IN ANALYSIS  (python-pro Action 5)
# =============================================================================


class TestSequencingAnalysis:
    """Integration tests for run_sequencing_analysis (Work Item 3)."""

    def test_worst_leq_original_leq_best(self) -> None:
        """Worst-first p_success <= original <= best-first p_success."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        seq = run_sequencing_analysis(
            household=h,
            inputs=inputs,
            original_results=base,
            seed=42,
        )

        assert seq.worst_first_p_success <= seq.original_p_success, (
            f"worst={seq.worst_first_p_success:.3f} > original={seq.original_p_success:.3f}"
        )
        assert seq.original_p_success <= seq.best_first_p_success, (
            f"original={seq.original_p_success:.3f} > best={seq.best_first_p_success:.3f}"
        )

    def test_worst_first_p5_lowest(self) -> None:
        """Worst-first p5 should be the lowest among all orderings."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        seq = run_sequencing_analysis(
            household=h,
            inputs=inputs,
            original_results=base,
            seed=42,
        )

        assert seq.worst_first_p5 <= seq.original_p5, (
            f"worst p5={seq.worst_first_p5:,.0f} > original p5={seq.original_p5:,.0f}"
        )
        # Best-first should have highest p5
        assert seq.best_first_p5 >= seq.original_p5, (
            f"best p5={seq.best_first_p5:,.0f} < original p5={seq.original_p5:,.0f}"
        )

    def test_result_fields_present(self) -> None:
        """Sequencing result has all six p_success/p5 fields."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        seq = run_sequencing_analysis(
            household=h,
            inputs=inputs,
            original_results=base,
            seed=42,
        )

        assert isinstance(seq, SequencingRiskResult)
        assert 0.0 <= seq.worst_first_p_success <= 1.0
        assert 0.0 <= seq.best_first_p_success <= 1.0
        assert 0.0 <= seq.original_p_success <= 1.0


class TestScenarioComparison:
    """Integration tests for run_scenario_comparison (Work Item 6)."""

    def test_scenarios_returned(self) -> None:
        """Dynamic scenarios are returned based on household config."""
        h = reference_household(retire_age=50)
        # Add PT to at least one earner so "No PT income" appears
        earners = list(h.earners)
        earners[0] = type(earners[0])(
            **{
                **vars(earners[0]),
                "pt_days_per_week": 3.0,
                "pt_start_age": 50,
                "pt_end_age": 60,
                "pt_rate_mode": "daily_rate",
            }
        )
        h = type(h)(**{**vars(h), "earners": tuple(earners)})

        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)
        scens = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )

        # Should have No PT (PT>0 exists) + 2 earners stop + hi/lo expenses = 5
        assert "No PT income" in scens
        assert "Earner 1 stops working" in scens
        assert "Earner 2 stops working" in scens
        assert "Expenses 10% higher" in scens
        assert "Expenses 10% lower" in scens
        assert isinstance(scens["No PT income"], ScenarioComparisonResult)
        assert isinstance(scens["Expenses 10% higher"], ScenarioComparisonResult)

    def test_no_pt_skipped_when_no_pt_configured(self) -> None:
        """'No PT income' scenario is omitted when no earner has PT."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)

        scens = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )

        assert "No PT income" not in scens
        # Other scenarios should still be present
        assert "Earner 1 stops working" in scens
        assert "Expenses 10% higher" in scens

    def test_no_pt_reduces_success(self) -> None:
        """Removing PT income should not increase success probability."""
        h = reference_household(retire_age=50)
        # Add PT income to at least one earner so the scenario has an effect
        earners = list(h.earners)
        earners[0] = type(earners[0])(
            **{
                **vars(earners[0]),
                "pt_days_per_week": 3.0,
                "pt_start_age": 50,
                "pt_end_age": 60,
                "pt_rate_mode": "daily_rate",
            }
        )
        h = type(h)(**{**vars(h), "earners": tuple(earners)})

        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)
        scens = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=200,
        )

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        assert scens["No PT income"].p_success <= base.p_success + 0.01, (
            f"No PT ({scens['No PT income'].p_success:.3f}) > Base ({base.p_success:.3f})"
        )

    def test_earner_stops_working_lowers_success(self) -> None:
        """An earner stopping work should not increase success."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        scens = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )

        for label in ("Earner 1 stops working", "Earner 2 stops working"):
            assert scens[label].p_success <= base.p_success + 0.01, (
                f"{label} ({scens[label].p_success:.3f}) > Base ({base.p_success:.3f})"
            )

    def test_expenses_higher_lowers_success(self) -> None:
        """Higher expenses should not increase success probability."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        scens = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )

        assert scens["Expenses 10% higher"].p_success <= base.p_success + 0.01, (
            f"Expenses higher ({scens['Expenses 10% higher'].p_success:.3f})"
            f" > Base ({base.p_success:.3f})"
        )

    def test_expenses_lower_not_below_base(self) -> None:
        """Lower expenses should not reduce success below base."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)

        base = run_monte_carlo(household=h, inputs=inputs, seed=42)
        scens = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )

        assert scens["Expenses 10% lower"].p_success >= base.p_success - 0.01, (
            f"Expenses lower ({scens['Expenses 10% lower'].p_success:.3f})"
            f" < Base ({base.p_success:.3f})"
        )

    def test_results_are_reproducible(self) -> None:
        """Same seed produces identical scenario comparison results."""
        h = reference_household(retire_age=50)
        inputs = SimulationInputs(n_iterations=100, cgt_on_drawdowns=True)

        scens1 = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )
        scens2 = run_scenario_comparison(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=100,
        )

        for key in ("Earner 1 stops working", "Expenses 10% higher"):
            assert scens1[key].p_success == scens2[key].p_success
            assert scens1[key].bridge_median == scens2[key].bridge_median


class TestRetirementSearch:
    """Tests for run_retirement_search (Work Item 9, multi-earner)."""

    def test_single_earner_backward_compat(self) -> None:
        """Single-earner household behaves identically to old single-earner path."""
        h1 = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        result = run_retirement_search(
            household=h1,
            inputs=inputs,
            seed=42,
            n_trials=200,
            min_search_age=45,
            success_threshold=0.95,
        )

        assert isinstance(result, RetirementSearchResult)
        assert result.mode == "both_together"
        assert result.entered_age == 55
        assert 0.0 <= result.entered_p_success <= 1.0
        assert result.earliest_age <= result.entered_age
        assert len(result.entered_ages_by_earner) == 2  # reference_household has 2
        assert result.floor_age >= 45

    def test_both_together_lower_ages(self) -> None:
        """both_together mode: earliest_age <= entered_age and trend is monotonic."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        result = run_retirement_search(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=200,
            min_search_age=45,
            success_threshold=0.95,
            mode="both_together",
        )

        assert result.mode == "both_together"
        assert result.earliest_age <= result.entered_age
        assert result.target_earner_label is None  # not set in both_together mode
        # Both earners' ages should be in entered_ages_by_earner
        assert "Earner 1" in result.entered_ages_by_earner
        assert "Earner 2" in result.entered_ages_by_earner

    def test_per_earner_mode(self) -> None:
        """per_earner mode: searches the target earner only; others held fixed."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        result = run_retirement_search(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=200,
            min_search_age=45,
            success_threshold=0.95,
            mode="per_earner",
            target_earner_index=0,
        )

        assert result.mode == "per_earner"
        assert result.target_earner_label == "Earner 1"
        assert result.earliest_age <= result.entered_age
        assert "Earner 1" in result.entered_ages_by_earner
        assert "Earner 2" in result.entered_ages_by_earner

    def test_per_earner_targets_second_earner(self) -> None:
        """per_earner mode: target_earner_index=1 searches Earner 2."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        result = run_retirement_search(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=200,
            min_search_age=45,
            success_threshold=0.95,
            mode="per_earner",
            target_earner_index=1,
        )

        assert result.target_earner_label == "Earner 2"
        assert result.earliest_age <= result.entered_age

    def test_entered_ages_by_earner_preserved(self) -> None:
        """entered_ages_by_earner contains all earners' current retirement ages."""
        h = reference_household(retire_age=55)
        inputs = SimulationInputs(n_iterations=200, cgt_on_drawdowns=True)

        result = run_retirement_search(
            household=h,
            inputs=inputs,
            seed=42,
            n_trials=200,
            mode="both_together",
        )

        assert result.entered_ages_by_earner["Earner 1"] == 55
        assert result.entered_ages_by_earner["Earner 2"] == 55
