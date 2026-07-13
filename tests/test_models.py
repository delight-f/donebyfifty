"""Tests for general-purpose data models.

Covers dataclass construction, property access, serialisation round-trips,
and backward compatibility with v1 profile format.
"""

from __future__ import annotations

import pytest

from models import (
    Child,
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
    Profile,
    SimulationInputs,
    SimulationResults,
    _upgrade_v1_profile,
)

# =============================================================================
# EARNER
# =============================================================================


class TestEarner:
    def test_defaults(self) -> None:
        e = Earner()
        assert e.label == "Earner 1"
        assert e.salary == 100_000.0
        assert e.super_balance == 100_000.0
        assert e.salary_growth_rate == 0.005  # real growth (above inflation)
        assert e.retirement_age == 50
        assert e.super_access_age == 60
        assert e.sg_rate == 0.12
        assert e.employment_type == "employed"
        assert e.is_employed is True
        assert e.pt_days_per_week == 0.0

    def test_immutable(self) -> None:
        e = Earner(salary=50_000.0)
        with pytest.raises(AttributeError):
            e.salary = 60_000.0  # type: ignore[misc]

    def test_non_working_earner(self) -> None:
        e = Earner(is_employed=False, salary=0.0, super_balance=50_000.0)
        assert e.is_employed is False
        assert e.salary == 0.0

    def test_part_time_params(self) -> None:
        e = Earner(
            pt_days_per_week=2.0,
            pt_start_age=55,
            pt_end_age=60,
            pt_daily_rate=500.0,
        )
        assert e.pt_days_per_week == 2.0
        assert e.pt_start_age == 55
        assert e.pt_end_age == 60
        assert e.pt_daily_rate == 500.0


# =============================================================================
# CHILD
# =============================================================================


class TestChild:
    def test_defaults(self) -> None:
        c = Child()
        assert c.label == "Child 1"
        assert c.age == 5
        assert c.education_schedule == ()

    def test_with_schedule(self) -> None:
        schedule = ((5, 10_000.0), (6, 10_500.0), (7, 11_000.0))
        c = Child(label="Alice", age=3, education_schedule=schedule)
        assert c.label == "Alice"
        assert c.age == 3
        assert len(c.education_schedule) == 3

    def test_empty_schedule_no_cost(self) -> None:
        c = Child(age=5)
        assert c.education_schedule == ()

    def test_immutable_schedule(self) -> None:
        c = Child(education_schedule=((5, 5000.0),))
        with pytest.raises(TypeError):
            c.education_schedule[0] = (6, 6000.0)  # type: ignore[index]


# =============================================================================
# MORTGAGE ACCOUNT
# =============================================================================


class TestMortgageAccount:
    def test_defaults(self) -> None:
        m = MortgageAccount()
        assert m.label == "Mortgage 1"
        assert m.principal == 500_000.0
        assert m.interest_rate == 0.0605
        assert m.monthly_payment == 0.0  # interest-only by default
        assert m.offset_accounts == ()

    def test_with_offset_link(self) -> None:
        m = MortgageAccount(
            principal=800_000.0,
            monthly_payment=5_000.0,
            offset_accounts=("Offset 1",),
        )
        assert m.principal == 800_000.0
        assert m.offset_accounts == ("Offset 1",)

    def test_interest_only_default(self) -> None:
        """Default monthly_payment=0 means interest-only loan."""
        m = MortgageAccount()
        assert m.monthly_payment == 0.0


# =============================================================================
# INVESTMENT ACCOUNT
# =============================================================================


class TestInvestmentAccount:
    def test_defaults(self) -> None:
        a = InvestmentAccount()
        assert a.label == "Account 1"
        assert a.market_value == 0.0
        assert a.cost_basis == 0.0
        assert a.asset_class == "equity"
        assert a.tax_jurisdiction == "au"
        assert a.cgt_rate == 0.30
        assert a.is_offset is False

    def test_offset_account(self) -> None:
        a = InvestmentAccount(
            label="My Offset",
            market_value=100_000.0,
            cost_basis=100_000.0,
            asset_class="cash",
            is_offset=True,
            cgt_rate=0.0,
        )
        assert a.is_offset is True
        assert a.cgt_rate == 0.0

    def test_has_gain(self) -> None:
        a = InvestmentAccount(market_value=150_000.0, cost_basis=100_000.0)
        assert a.has_gain is True

    def test_no_gain(self) -> None:
        a = InvestmentAccount(market_value=80_000.0, cost_basis=100_000.0)
        assert a.has_gain is False

    def test_exact_basis(self) -> None:
        a = InvestmentAccount(market_value=100_000.0, cost_basis=100_000.0)
        assert a.has_gain is False

    def test_uk_jurisdiction(self) -> None:
        a = InvestmentAccount(label="UK ETFs", tax_jurisdiction="uk", cgt_rate=0.30)
        assert a.tax_jurisdiction == "uk"


# =============================================================================
# HOUSEHOLD
# =============================================================================


class TestHousehold:
    def test_default_single_earner(self) -> None:
        h = Household()
        assert h.num_earners == 1
        assert h.num_children == 0
        assert h.num_mortgages == 0
        assert h.num_investment_accounts == 0
        assert h.base_living_expenses == 60_000.0
        assert h.retirement_target == 80_000.0

    def test_multi_earner_household(self) -> None:
        e1 = Earner(label="Person A", salary=120_000.0, super_balance=80_000.0)
        e2 = Earner(label="Person B", salary=80_000.0, super_balance=40_000.0)
        h = Household(earners=(e1, e2))
        assert h.num_earners == 2
        assert h.total_super == 120_000.0

    def test_family_with_children(self) -> None:
        e1 = Earner(label="Parent 1")
        e2 = Earner(label="Parent 2")
        c1 = Child(label="Child A", age=5)
        c2 = Child(label="Child B", age=8)
        c3 = Child(label="Child C", age=12)
        h = Household(earners=(e1, e2), children=(c1, c2, c3))
        assert h.num_children == 3
        assert h.num_earners == 2

    def test_with_mortgage_and_accounts(self) -> None:
        m = MortgageAccount(principal=600_000.0, monthly_payment=4_000.0)
        offset = InvestmentAccount(
            label="Offset", market_value=50_000.0, cost_basis=50_000.0, is_offset=True
        )
        equities = InvestmentAccount(label="Shares", market_value=200_000.0, cost_basis=150_000.0)
        h = Household(
            mortgages=(m,),
            investment_accounts=(offset, equities),
        )
        assert h.num_mortgages == 1
        assert h.num_investment_accounts == 2
        assert h.total_mortgage_principal == 600_000.0
        assert h.total_bridge_assets == 200_000.0  # only equities
        assert h.total_offset_balance == 50_000.0

    def test_min_retirement_age(self) -> None:
        e1 = Earner(retirement_age=55)
        e2 = Earner(retirement_age=60)
        h = Household(earners=(e1, e2))
        assert h.min_retirement_age == 55

    def test_min_super_access_age(self) -> None:
        e1 = Earner(super_access_age=60)
        e2 = Earner(super_access_age=55)
        h = Household(earners=(e1, e2))
        assert h.min_super_access_age == 55

    def test_empty_earners(self) -> None:
        """Should not crash with zero earners (edge case)."""
        h = Household(earners=())
        assert h.num_earners == 0
        assert h.min_retirement_age == 999
        assert h.min_super_access_age == 999


# =============================================================================
# SIMULATION INPUTS
# =============================================================================


class TestSimulationInputs:
    def test_defaults(self) -> None:
        s = SimulationInputs()
        assert s.n_iterations == 5_000
        assert s.inflation == 0.025
        assert s.simulation_start_age == 37
        assert s.simulation_end_age == 72
        assert s.cgt_on_drawdowns is True
        assert s.sell_strategy == "waterfall"
        assert s.sell_order == ()

    def test_with_household(self) -> None:
        h = Household(base_living_expenses=80_000.0)
        s = SimulationInputs(household=h, n_iterations=10_000)
        assert s.household.base_living_expenses == 80_000.0
        assert s.n_iterations == 10_000


# =============================================================================
# SIMULATION RESULTS
# =============================================================================


class TestSimulationResults:
    def test_minimal(self) -> None:
        r = SimulationResults(
            trials=100,
            p_success=0.85,
            bridge_mean=500_000.0,
            bridge_median=480_000.0,
            bridge_p5=100_000.0,
            bridge_p10=200_000.0,
            bridge_p25=300_000.0,
            bridge_p75=700_000.0,
            bridge_p90=900_000.0,
            bridge_p95=1_000_000.0,
            bridge_min=50_000.0,
            super_median=600_000.0,
            horizon_age=60,
            per_earner_super_p50={"Person A": 350_000.0, "Person B": 250_000.0},
            remaining_mortgage_p50={"Mortgage 1": 0.0},
        )
        assert r.p_success == 0.85
        assert r.per_earner_super_p50["Person A"] == 350_000.0

    def test_summary_dict_roundtrip(self) -> None:
        r1 = SimulationResults(
            trials=5000,
            p_success=0.92,
            bridge_mean=600_000.0,
            bridge_median=580_000.0,
            bridge_p5=200_000.0,
            bridge_p10=300_000.0,
            bridge_p25=400_000.0,
            bridge_p75=800_000.0,
            bridge_p90=1_000_000.0,
            bridge_p95=1_200_000.0,
            bridge_min=100_000.0,
            super_median=700_000.0,
            horizon_age=60,
            per_earner_super_p50={"A": 400_000.0},
        )
        d = r1.summary_dict()
        r2 = SimulationResults.from_dict(d)
        assert r1 == r2

    def test_summary_dict_defaults(self) -> None:
        r1 = SimulationResults(
            trials=100,
            p_success=0.5,
            bridge_mean=0.0,
            bridge_median=0.0,
            bridge_p5=0.0,
            bridge_p10=0.0,
            bridge_p25=0.0,
            bridge_p75=0.0,
            bridge_p90=0.0,
            bridge_p95=0.0,
            bridge_min=0.0,
            super_median=0.0,
        )
        d = r1.summary_dict()
        # Default horizon_age should be 60
        assert d["horizon_age"] == 60
        assert d["per_earner_super_p50"] == {}


# =============================================================================
# PROFILE SERIALISATION
# =============================================================================


class TestProfileSerialisation:
    def test_roundtrip_basic(self) -> None:
        inputs = SimulationInputs(n_iterations=2_000)
        results = SimulationResults(
            trials=2000,
            p_success=0.88,
            bridge_mean=400_000.0,
            bridge_median=390_000.0,
            bridge_p5=100_000.0,
            bridge_p10=200_000.0,
            bridge_p25=280_000.0,
            bridge_p75=520_000.0,
            bridge_p90=650_000.0,
            bridge_p95=750_000.0,
            bridge_min=50_000.0,
            super_median=500_000.0,
        )
        p1 = Profile(
            profile_name="Test Profile",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00",
            inputs=inputs,
            last_results=results,
        )
        d = p1.to_dict()
        p2 = Profile.from_dict(d)
        assert p2.profile_name == "Test Profile"
        assert p2.created_at == "2024-01-01T00:00:00"
        assert p2.inputs.n_iterations == 2_000
        assert p2.last_results is not None
        assert p2.last_results.p_success == 0.88

    def test_roundtrip_no_results(self) -> None:
        p1 = Profile(profile_name="Empty")
        d = p1.to_dict()
        p2 = Profile.from_dict(d)
        assert p2.profile_name == "Empty"
        assert p2.last_results is None

    def test_roundtrip_multi_child(self) -> None:
        e1 = Earner(label="Parent")
        c1 = Child(label="Child A", age=5)
        c2 = Child(label="Child B", age=8)
        c3 = Child(label="Child C", age=12)
        h = Household(
            earners=(e1,),
            children=(c1, c2, c3),
            base_living_expenses=70_000.0,
            retirement_target=90_000.0,
        )
        inputs = SimulationInputs(household=h)
        p1 = Profile(profile_name="Big Family", inputs=inputs)
        d = p1.to_dict()
        p2 = Profile.from_dict(d)
        assert p2.inputs.household.num_children == 3
        assert p2.inputs.household.base_living_expenses == 70_000.0
        assert p2.inputs.household.children[1].age == 8

    def test_roundtrip_multiple_mortgages(self) -> None:
        m1 = MortgageAccount(label="Home Loan", principal=600_000.0)
        m2 = MortgageAccount(label="IP Loan", principal=400_000.0)
        h = Household(mortgages=(m1, m2))
        inputs = SimulationInputs(household=h)
        p1 = Profile(profile_name="Two Mortgages", inputs=inputs)
        d = p1.to_dict()
        p2 = Profile.from_dict(d)
        assert p2.inputs.household.num_mortgages == 2
        assert p2.inputs.household.mortgages[1].principal == 400_000.0

    def test_roundtrip_investment_accounts(self) -> None:
        a1 = InvestmentAccount(label="Shares", market_value=200_000.0, cost_basis=100_000.0)
        a2 = InvestmentAccount(
            label="Offset", market_value=50_000.0, cost_basis=50_000.0, is_offset=True
        )
        h = Household(investment_accounts=(a1, a2))
        inputs = SimulationInputs(household=h)
        p1 = Profile(profile_name="Two Accounts", inputs=inputs)
        d = p1.to_dict()
        p2 = Profile.from_dict(d)
        assert p2.inputs.household.num_investment_accounts == 2
        assert p2.inputs.household.investment_accounts[1].is_offset is True

    def test_roundtrip_with_schedule(self) -> None:
        schedule = ((5, 10_000.0), (6, 11_000.0), (7, 12_000.0))
        c = Child(label="Scholar", age=3, education_schedule=schedule)
        h = Household(children=(c,))
        inputs = SimulationInputs(household=h)
        p1 = Profile(profile_name="With Schedule", inputs=inputs)
        d = p1.to_dict()
        p2 = Profile.from_dict(d)
        assert len(p2.inputs.household.children[0].education_schedule) == 3
        age, cost = p2.inputs.household.children[0].education_schedule[1]
        assert age == 6
        assert cost == 11_000.0

    def test_version_marker(self) -> None:
        p = Profile(profile_name="V2")
        d = p.to_dict()
        assert d["_version"] == 2


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


class TestBackwardCompat:
    def test_v1_to_v2_upgrade(self) -> None:
        """A v1-format profile dict should upgrade to v2 without error."""
        v1_data = {
            "profile_name": "Legacy v1",
            "created_at": "2024-06-01T00:00:00",
            "updated_at": "2024-06-15T00:00:00",
            "inputs": {
                "retire_age": 50,
                "n_iterations": 5000,
                "inflation": 0.025,
                "cgt_on_drawdowns": True,
                "sell_uk": False,
                "pt_days_per_week": 0.5,
                "pt_start_age": 50,
                "pt_end_age": 55,
                "extra_cash": 0.0,
                "finances": {
                    "salary_h": 320_000.0,
                    "salary_w": 200_000.0,
                    "mortgage": 820_000.0,
                    "offset": 200_000.0,
                    "uk_etfs": 512_000.0,
                    "uk_basis": 201_000.0,
                    "au_etfs": 63_000.0,
                    "au_basis": 63_000.0,
                    "super_h": 310_720.71,
                    "super_w": 238_380.63,
                    "living_expenses": 75_000.0,
                    "child_age": 2,
                },
            },
            "last_results": {
                "trials": 5000,
                "p_success": 0.95,
                "bridge_mean": 600_000.0,
                "bridge_median": 580_000.0,
                "bridge_p5": 200_000.0,
                "bridge_p10": 300_000.0,
                "bridge_p25": 400_000.0,
                "bridge_p75": 800_000.0,
                "bridge_p90": 1_000_000.0,
                "bridge_p95": 1_200_000.0,
                "bridge_min": 100_000.0,
                "super_median": 700_000.0,
            },
        }
        upgraded = _upgrade_v1_profile(v1_data)
        assert upgraded["_version"] == 2
        assert len(upgraded["inputs"]["household"]["earners"]) == 2
        assert len(upgraded["inputs"]["household"]["children"]) == 1
        assert len(upgraded["inputs"]["household"]["mortgages"]) == 1
        assert len(upgraded["inputs"]["household"]["investment_accounts"]) == 3  # offset + UK + AU

    def test_v2_already_upgraded(self) -> None:
        """Already versioned dict should pass through unchanged."""
        data = {
            "_version": 2,
            "profile_name": "Already v2",
            "inputs": {
                "household": {
                    "earners": [{"label": "Me", "salary": 100_000.0}],
                    "children": [],
                    "mortgages": [],
                    "investment_accounts": [],
                    "base_living_expenses": 60_000.0,
                    "retirement_target": 80_000.0,
                }
            },
        }
        upgraded = _upgrade_v1_profile(data)
        assert upgraded["_version"] == 2
        assert upgraded["profile_name"] == "Already v2"

    def test_v1_upgrade_compatible_profile_load(self) -> None:
        """Full round-trip: v1 dict -> Profile.from_dict -> valid Profile."""
        v1_data = {
            "profile_name": "Legacy",
            "inputs": {
                "retire_age": 50,
                "n_iterations": 5000,
                "inflation": 0.025,
                "cgt_on_drawdowns": True,
                "sell_uk": False,
                "pt_days_per_week": 0.0,
                "pt_start_age": 50,
                "pt_end_age": 52,
                "extra_cash": 0.0,
                "finances": {
                    "salary_h": 320_000.0,
                    "salary_w": 200_000.0,
                    "mortgage": 820_000.0,
                    "offset": 200_000.0,
                    "uk_etfs": 512_000.0,
                    "uk_basis": 201_000.0,
                    "au_etfs": 63_000.0,
                    "au_basis": 63_000.0,
                    "super_h": 310_720.71,
                    "super_w": 238_380.63,
                    "living_expenses": 75_000.0,
                    "child_age": 2,
                },
            },
            "last_results": None,
        }
        profile = Profile.from_dict(v1_data)
        assert profile.profile_name == "Legacy"
        assert profile.inputs.household.num_earners == 2
        assert profile.inputs.household.num_children == 1
        assert profile.inputs.household.num_mortgages == 1
        assert profile.inputs.household.investment_accounts[0].is_offset is True
        assert profile.last_results is None

    def test_v1_no_finances(self) -> None:
        """Dict with no finances key should pass through as-is."""
        data = {"profile_name": "Empty", "inputs": {}}
        upgraded = _upgrade_v1_profile(data)
        assert upgraded["profile_name"] == "Empty"
