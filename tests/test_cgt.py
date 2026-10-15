"""Hand-calculated tests for the CGT ledger, sale solver and investment-income tax (D12-D14).

Functions take arrays with a leading trial axis (D22). Scalar cases wrap their inputs in
one-element arrays and read element 0. The batch tests check that each trial's result is the
same whether it is computed alone or in a batch.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from donebyfifty.cgt import (
    Ledger,
    Positions,
    TaxContext,
    add_cost,
    cost_total,
    investment_income_tax,
    owner_tax,
    realised_gains,
    reform_snapshot,
    sale_tax,
    scale,
    solve_sale,
    stack_income,
)
from donebyfifty.tax import brackets_for_fy

FloatArray = NDArray[np.float64]

FY26 = brackets_for_fy(2026)
YEARS = 9  # price index has YEARS + 1 entries
FLAT_PRICE = np.ones((1, YEARS + 1))
SOLE = [(0, 1.0)]


def no_surcharge(family_mls: FloatArray) -> FloatArray:
    """MLS surcharge of zero for every trial."""
    return np.zeros_like(family_mls)


def couple_surcharge(family_mls: FloatArray) -> FloatArray:
    """1.25% couple MLS tier at $246k-$328k family MLS income, else zero."""
    return np.where((family_mls >= 246_000.0) & (family_mls < 328_000.0), 0.0125, 0.0)


CTX = TaxContext(brackets=FY26, dependants=0, couple=False, surcharge=no_surcharge)
COUPLE_CTX = TaxContext(brackets=FY26, dependants=0, couple=True, surcharge=no_surcharge)
SOLE_PARENT_CTX = TaxContext(brackets=FY26, dependants=1, couple=False, surcharge=no_surcharge)
MLS_CTX = TaxContext(brackets=FY26, dependants=0, couple=True, surcharge=couple_surcharge)

# Reference tax values (FY26 brackets, single owner, no surcharge). Bracket tax is
# 15% on 18,201-45,000, 30% on 45,001-135,000, 37% on 135,001-190,000. Medicare is 2%.
# LITO is 325 - 1.5% * (income - 45,000) for incomes 45,000-66,667, and 0 above that.
#   tax(10_000) = 0     (below the $18,200 tax-free threshold)
#   tax(50_000) = 4,020 + 1,500 + 1,000 - LITO 250 = 6,270
#   tax(60_000) = 4,020 + 4,500 + 1,200 - LITO 100 = 9,620
#   tax(100_000) = 4,020 + 16,500 + 2,000 = 22,520 (no LITO)


def one(value: float) -> FloatArray:
    """A one-trial array."""
    return np.array([value])


def snapped_ledger(cost: float, pre_reform_gain: float, years: int = YEARS) -> Ledger:
    """One-trial, one-account ledger snapped at year 0 with the given basis and gain."""
    ledger = Ledger.empty(1, 1, years)
    ledger.cost[0, 0, 0] = cost
    ledger.pre_reform_gain[0, 0] = pre_reform_gain
    ledger.snapped = True
    return ledger


def trial_ledger(ledger: Ledger, trial: int) -> Ledger:
    """Single-trial copy of ``ledger`` for trial ``trial``."""
    return Ledger(
        cost=ledger.cost[trial : trial + 1].copy(),
        pre_reform_gain=ledger.pre_reform_gain[trial : trial + 1].copy(),
        snapped=ledger.snapped,
    )


class TestLedger:
    """Cost bases and the reform snapshot (D13)."""

    def test_opening_basis_is_booked_in_year_zero(self) -> None:
        ledger = Ledger.empty(1, 1, YEARS)
        add_cost(ledger, 0, one(100_000.0), 0)
        assert ledger.cost[0, 0, 0] == 100_000.0
        assert cost_total(ledger)[0, 0] == 100_000.0
        assert not ledger.snapped
        assert cost_total(Ledger.empty(1, 1, YEARS))[0, 0] == 0.0

    def test_add_cost_ignores_non_positive_amounts(self) -> None:
        ledger = Ledger.empty(1, 1, YEARS)
        add_cost(ledger, 0, one(0.0), 3)
        add_cost(ledger, 0, one(-5.0), 3)
        assert cost_total(ledger)[0, 0] == 0.0
        add_cost(ledger, 0, one(500.0), 3)
        assert ledger.cost[0, 0, 3] == 500.0
        assert cost_total(ledger)[0, 0] == 500.0

    def test_reform_snapshot_freezes_gain_and_resets_cost_to_market(self) -> None:
        # Bought for 100k, worth 300k at the reform: pre-reform gain 200k, new cost 300k.
        ledger = Ledger.empty(1, 1, YEARS)
        add_cost(ledger, 0, one(100_000.0), 0)
        reform_snapshot(ledger, np.array([[300_000.0]]), 2)
        assert ledger.snapped
        assert ledger.pre_reform_gain[0, 0] == 200_000.0
        assert ledger.cost[0, 0, 0] == 0.0
        assert ledger.cost[0, 0, 2] == 300_000.0
        assert cost_total(ledger)[0, 0] == 300_000.0

    def test_scale_keeps_proportion_of_cost_and_gain(self) -> None:
        ledger = snapped_ledger(200_000.0, 100_000.0)
        scale(ledger, 0, one(0.25))
        assert cost_total(ledger)[0, 0] == pytest.approx(50_000.0)
        assert ledger.pre_reform_gain[0, 0] == pytest.approx(25_000.0)

    def test_post_reform_deposit_is_not_taxed(self) -> None:
        # C2: after the snapshot a 50k deposit adds 50k of cost. Selling everything at
        # 350k realises 350k - (300k + 50k) = 0 post-reform gain; the 200k pre-reform
        # gain is still the only discountable amount.
        ledger = Ledger.empty(1, 1, YEARS)
        add_cost(ledger, 0, one(100_000.0), 0)
        reform_snapshot(ledger, np.array([[300_000.0]]), 0)
        add_cost(ledger, 0, one(50_000.0), 5)
        discountable, post = realised_gains(ledger, 0, one(350_000.0), one(1.0), 5, FLAT_PRICE)
        assert discountable[0] == 200_000.0
        assert post[0] == 0.0

    def test_post_reform_cost_is_cpi_indexed(self) -> None:
        # 100k incurred in year 0; price index 1.21 in year 2 -> indexed cost 121k.
        # Value 200k -> post-reform gain 79k, no discountable part.
        ledger = snapped_ledger(100_000.0, 0.0, years=2)
        price_index = np.array([[1.0, 1.1, 1.21]])
        discountable, post = realised_gains(ledger, 0, one(200_000.0), one(1.0), 2, price_index)
        assert discountable[0] == 0.0
        assert post[0] == pytest.approx(79_000.0)


class TestSaleTax:
    """Per-owner CGT on a sale (D13)."""

    def test_pre_reform_gain_is_discounted_and_not_floored(self) -> None:
        # Snapshot state: cost reset to the 200k market value, pre-reform gain 100k.
        # Discount halves it to 50k taxable; tax(50k) = 6,270. No post-reform gain, so
        # the 30% floor does not apply.
        ledger = snapped_ledger(200_000.0, 100_000.0)
        sale = sale_tax(
            CTX, Positions.zero(1, 1), ledger, 0, SOLE, one(200_000.0), one(1.0), 0, FLAT_PRICE
        )
        assert sale.tax[0] == pytest.approx(6_270.0)
        assert sale.tax_without_floor[0] == pytest.approx(6_270.0)

    def test_floor_binds_on_low_post_reform_gain(self) -> None:
        # Post-reform gain 100k. Stacked tax(100k) = 22,520, but the 30% floor gives
        # 0.30 * 100k = 30,000, which is what is payable.
        ledger = snapped_ledger(100_000.0, 0.0)
        sale = sale_tax(
            CTX, Positions.zero(1, 1), ledger, 0, SOLE, one(200_000.0), one(1.0), 0, FLAT_PRICE
        )
        assert sale.tax[0] == pytest.approx(30_000.0)
        assert sale.tax_without_floor[0] == pytest.approx(22_520.0)

    def test_capital_loss_carries_forward_and_offsets_later_gain(self) -> None:
        positions = Positions.zero(1, 1)
        # Loss: 300k cost, 200k value -> -100k. No tax; 100k goes to the loss pool.
        loss_ledger = snapped_ledger(300_000.0, 0.0)
        loss_sale = sale_tax(
            CTX, positions, loss_ledger, 0, SOLE, one(200_000.0), one(1.0), 0, FLAT_PRICE
        )
        assert loss_sale.tax[0] == 0.0
        assert positions.loss_pool[0, 0] == pytest.approx(100_000.0)
        # Gain: 100k cost, 250k value -> +150k. The 100k pool offsets it -> 50k left.
        # Post-reform 50k: stacked tax(50k) = 6,270, floor 0.30 * 50k = 15,000 -> 15,000.
        gain_ledger = snapped_ledger(100_000.0, 0.0)
        gain_sale = sale_tax(
            CTX, positions, gain_ledger, 0, SOLE, one(250_000.0), one(1.0), 0, FLAT_PRICE
        )
        assert gain_sale.tax[0] == pytest.approx(15_000.0)
        assert gain_sale.tax_without_floor[0] == pytest.approx(6_270.0)
        assert positions.loss_pool[0, 0] == 0.0


class TestSolveSale:
    """Gross-up of a sale so that proceeds net of CGT cover a need."""

    def test_retiree_60k_from_1m_with_200k_basis(self) -> None:
        # C4: 1,000,000 value, 200,000 basis, post-reform, need 60,000 net.
        # Gain per unit sold = 800,000. The floor binds at these gains, so tax = 0.30 * gain.
        # Net = 1,000,000 * f - 0.30 * 800,000 * f = 760,000 * f = 60,000
        #   -> f = 60,000 / 760,000 = 0.078947368..., proceeds 78,947.37, tax 18,947.37.
        # Check the floor binds: stacked tax(63,157.89) = 4,020 + 0.30 * 18,157.89
        #   + 2% * 63,157.89 - LITO 52.63 = 10,678 < 18,947.37.
        ledger = snapped_ledger(200_000.0, 0.0)
        fraction, sale = solve_sale(
            CTX,
            Positions.zero(1, 1),
            ledger,
            0,
            SOLE,
            one(1_000_000.0),
            one(60_000.0),
            0,
            FLAT_PRICE,
        )
        assert fraction[0] == pytest.approx(60_000.0 / 760_000.0, rel=1e-6)
        assert sale.tax[0] == pytest.approx(18_947.37, abs=0.05)
        assert fraction[0] * 1_000_000.0 - sale.tax[0] == pytest.approx(60_000.0, abs=0.05)

    def test_sells_everything_when_need_exceeds_proceeds(self) -> None:
        # No gain (cost equals value) -> no tax; proceeds 100k < need 1M -> sell all.
        ledger = snapped_ledger(100_000.0, 0.0)
        fraction, sale = solve_sale(
            CTX,
            Positions.zero(1, 1),
            ledger,
            0,
            SOLE,
            one(100_000.0),
            one(1_000_000.0),
            0,
            FLAT_PRICE,
        )
        assert fraction[0] == 1.0
        assert sale.tax[0] == 0.0

    def test_tax_never_exceeds_proceeds(self) -> None:
        # Zero-cost holding: every dollar is gain. Tax is always below proceeds (rate < 100%).
        ledger = snapped_ledger(0.0, 0.0)
        fraction, sale = solve_sale(
            CTX,
            Positions.zero(1, 1),
            ledger,
            0,
            SOLE,
            one(1_000_000.0),
            one(500_000.0),
            0,
            FLAT_PRICE,
        )
        assert 0.0 < fraction[0] < 1.0
        assert sale.tax[0] < fraction[0] * 1_000_000.0

    def test_zero_value_holding_returns_nothing(self) -> None:
        ledger = snapped_ledger(0.0, 0.0)
        fraction, sale = solve_sale(
            CTX, Positions.zero(1, 1), ledger, 0, SOLE, one(0.0), one(10_000.0), 0, FLAT_PRICE
        )
        assert fraction[0] == 0.0
        assert sale.tax[0] == 0.0
        assert sale.tax_without_floor[0] == 0.0


class TestPositionStacking:
    """Income slices stack on an owner's position (D12)."""

    def test_two_slices_sum_to_combined_tax(self) -> None:
        # 60k then 40k on the same owner: tax(60k) = 9,620, then
        # tax(100k) - tax(60k) = 22,520 - 9,620 = 12,900. Sum = tax(100k) = 22,520.
        positions = Positions.zero(1, 1)
        first = stack_income(CTX, positions, 0, one(60_000.0))[0]
        second = stack_income(CTX, positions, 0, one(40_000.0))[0]
        assert first == pytest.approx(9_620.0, abs=0.01)
        assert second == pytest.approx(12_900.0, abs=0.01)
        assert first + second == pytest.approx(22_520.0, abs=1.0)

    def test_slice_updates_taxable_and_mls_totals_separately(self) -> None:
        # Owner 0: taxable 40,000 with 20,000 sacrifice (MLS income 60,000); owner 1: taxable
        # 10,000 (MLS 10,000). Family taxable 50,000, family MLS 70,000. Before the slice:
        # owner_tax(40,000, 50,000, 70,000) = 2,915.96 (see TestOwnerTax; surcharge is 0).
        # Stack 10,000 on owner 0: taxable 50,000, family taxable 60,000, family MLS 80,000.
        # Income tax(50,000) = 5,520 - LITO 250 = 5,270. Medicare family reduction =
        # min(full 1,000, 10% * (60,000 - 47,238) * 50,000 / 60,000 = 1,063.50) = 1,000.
        # After = 6,270, so the increment = 6,270 - 2,915.96 = 3,354.04.
        positions = Positions(
            np.array([[40_000.0, 10_000.0]]),
            np.array([[60_000.0, 10_000.0]]),
            np.zeros((1, 2)),
        )
        increment = stack_income(COUPLE_CTX, positions, 0, one(10_000.0))[0]
        assert increment == pytest.approx(3_354.04, abs=0.01)
        assert positions.assessable.tolist() == [[50_000.0, 10_000.0]]
        assert positions.mls_income.tolist() == [[70_000.0, 10_000.0]]


class TestOwnerTax:
    """Medicare family test and MLS income for one owner (taxable vs MLS income)."""

    def test_single_without_children_gets_individual_reduction_only(self) -> None:
        # Taxable 32,000, family values ignored because a single without children has no
        # family test. Bracket tax 15% * (32,000 - 18,200) = 2,070; LITO 700 (full below
        # 37,500), so income tax = 1,370. Individual Medicare reduction =
        # 10% * (32,000 - 28,011) = 398.90, below the 2% full levy of 640.
        # Total = 1,370 + 398.90 = 1,768.90. Passing family values of 100,000 must not
        # change this (the family test would give a 640 levy and total 2,010).
        tax_owed = owner_tax(CTX, one(32_000.0), one(100_000.0), one(100_000.0))
        assert tax_owed[0] == pytest.approx(1_768.90, abs=0.01)

    def test_couple_family_reduction_uses_family_taxable_income(self) -> None:
        # Owner taxable 40,000; family taxable 50,000 (other owner 10,000). Income tax:
        # bracket 15% * (40,000 - 18,200) = 3,270; LITO 700 - 5% * (40,000 - 37,500) = 575;
        # net 2,695. Family reduction = 10% * (50,000 - 47,238) * 40,000 / 50,000 = 220.96,
        # below the 800 full levy. Total = 2,695 + 220.96 = 2,915.96. Family MLS income
        # 70,000 (owner sacrifices 20,000) has no effect here: MLS surcharge is 0.
        tax_owed = owner_tax(COUPLE_CTX, one(40_000.0), one(50_000.0), one(70_000.0))
        assert tax_owed[0] == pytest.approx(2_915.96, abs=0.01)

    def test_sole_parent_family_test_uses_own_income(self) -> None:
        # One dependant: family lower threshold = 47,238 + 4,338 = 51,576. Own taxable
        # 40,000 is below it, so the Medicare levy is 0 and total = income tax 2,695.
        tax_owed = owner_tax(SOLE_PARENT_CTX, one(40_000.0), one(40_000.0), one(40_000.0))
        assert tax_owed[0] == pytest.approx(2_695.0, abs=0.01)

    def test_sacrifice_counts_for_mls_but_not_for_medicare_family_test(self) -> None:
        # Owner A: taxable 150,000 after 80,000 sacrifice (MLS income 230,000); owner B:
        # taxable 40,000. Family taxable 190,000; family MLS income 270,000, which is in
        # the 1.25% tier ($246k-$328k, couple, no children). If MLS were tested on taxable
        # income (190,000 < $210k) the surcharge would be 0.
        # Income tax(150,000) = 4,020 + 27,000 + 0.37 * 15,000 = 36,570 (no LITO).
        # Medicare family: 10% * (190,000 - 47,238) * 150,000 / 190,000 = 11,271 > full
        # levy 3,000, so 3,000. MLS = 1.25% * 150,000 = 1,875 (surcharge on taxable income).
        # Total = 36,570 + 3,000 + 1,875 = 41,445.
        tax_owed = owner_tax(MLS_CTX, one(150_000.0), one(190_000.0), one(270_000.0))
        assert tax_owed[0] == pytest.approx(41_445.0, abs=0.01)


class TestInvestmentIncomeTax:
    """Net tax on distributions with franking credits (D14)."""

    def test_franking_credit_refunds_tax_on_low_income(self) -> None:
        # Franked 7,000 grosses up to 7,000 / 0.7 = 10,000 (tax 0). Credit =
        # 7,000 * (1/0.7 - 1) = 3,000, refundable -> tax = -3,000.
        tax_owed = investment_income_tax(CTX, Positions.zero(1, 1), SOLE, one(0.0), one(7_000.0))
        assert tax_owed[0] == pytest.approx(-3_000.0)

    def test_unfranked_income_taxed_at_marginal_rates(self) -> None:
        # Unfranked 60,000 is taxed in full: tax(60k) = 9,620.
        tax_owed = investment_income_tax(CTX, Positions.zero(1, 1), SOLE, one(60_000.0), one(0.0))
        assert tax_owed[0] == pytest.approx(9_620.0)


class TestBatchEquivalence:
    """A batch gives each trial the same result it gets on its own (D22)."""

    def test_owner_tax_batch_matches_single_trials(self) -> None:
        taxable = np.array([32_000.0, 40_000.0, 150_000.0])
        family_taxable = np.array([100_000.0, 50_000.0, 190_000.0])
        family_mls = np.array([100_000.0, 70_000.0, 270_000.0])
        batch = owner_tax(MLS_CTX, taxable, family_taxable, family_mls)
        for t in range(3):
            single = owner_tax(
                MLS_CTX, taxable[t : t + 1], family_taxable[t : t + 1], family_mls[t : t + 1]
            )
            assert batch[t] == pytest.approx(single[0], abs=1e-9)

    def test_solve_sale_batch_matches_single_trials(self) -> None:
        market = np.array([1_000_000.0, 300_000.0, 100_000.0])
        need = np.array([60_000.0, 50_000.0, 1_000_000.0])
        ledger = Ledger.empty(3, 1, YEARS)
        ledger.cost[:, 0, 0] = np.array([200_000.0, 250_000.0, 100_000.0])
        ledger.pre_reform_gain[:, 0] = np.array([0.0, 40_000.0, 0.0])
        ledger.snapped = True
        prices = np.ones((3, YEARS + 1))
        fraction, sale = solve_sale(
            CTX, Positions.zero(3, 1), ledger, 0, SOLE, market, need, 0, prices
        )
        for t in range(3):
            single_fraction, single_sale = solve_sale(
                CTX,
                Positions.zero(1, 1),
                trial_ledger(ledger, t),
                0,
                SOLE,
                market[t : t + 1],
                need[t : t + 1],
                0,
                prices[t : t + 1],
            )
            assert fraction[t] == pytest.approx(single_fraction[0], abs=1e-12)
            assert sale.tax[t] == pytest.approx(single_sale.tax[0], abs=1e-6)
            assert sale.tax_without_floor[t] == pytest.approx(
                single_sale.tax_without_floor[0], abs=1e-6
            )

    def test_sale_tax_batch_matches_single_trials_with_loss_pool(self) -> None:
        # Two owners share the holding 50/50 and carry different loss pools, so every
        # trial takes a different path through the loss and floor logic.
        ownership = [(0, 0.5), (1, 0.5)]
        market = np.array([250_000.0, 200_000.0])
        fraction = np.array([1.0, 1.0])
        ledger = Ledger.empty(2, 1, YEARS)
        ledger.cost[:, 0, 0] = np.array([100_000.0, 300_000.0])
        ledger.snapped = True
        assessable = np.array([[40_000.0, 10_000.0], [0.0, 0.0]])
        mls = np.array([[60_000.0, 10_000.0], [0.0, 0.0]])
        pool = np.array([[100_000.0, 0.0], [0.0, 50_000.0]])
        positions = Positions(assessable.copy(), mls.copy(), pool.copy())
        sale = sale_tax(
            MLS_CTX, positions, ledger, 0, ownership, market, fraction, 0, np.ones((2, YEARS + 1))
        )
        for t in range(2):
            single_positions = Positions(
                assessable[t : t + 1].copy(), mls[t : t + 1].copy(), pool[t : t + 1].copy()
            )
            single = sale_tax(
                MLS_CTX,
                single_positions,
                trial_ledger(ledger, t),
                0,
                ownership,
                market[t : t + 1],
                fraction[t : t + 1],
                0,
                np.ones((1, YEARS + 1)),
            )
            assert sale.tax[t] == pytest.approx(single.tax[0], abs=1e-6)
            assert sale.tax_without_floor[t] == pytest.approx(single.tax_without_floor[0], abs=1e-6)
            assert positions.loss_pool[t].tolist() == pytest.approx(
                single_positions.loss_pool[0].tolist()
            )
