"""Shared pytest configuration and the reference household builder.

The package is importable through ``pythonpath = ["src"]`` in pyproject.toml.
Test modules import ``reference_household`` with ``from conftest import reference_household``.
"""

import random

import pytest

from donebyfifty.config import DEFAULT_EDU_SCHEDULE
from donebyfifty.models import (
    Child,
    Earner,
    Household,
    InvestmentAccount,
    MortgageAccount,
)


@pytest.fixture(autouse=True)
def _seed_module_rng() -> None:
    """Seed the stdlib ``random`` module before every test.

    The only consumer is ``main.py``, which draws a default seed with ``random.randint``
    when the user gives none. Seeding keeps that choice reproducible in tests. The
    simulation itself takes explicit seeds and never reads this module.
    """
    random.seed(12345)


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
        education_schedule=DEFAULT_EDU_SCHEDULE,
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
            )
        )
        accounts.append(
            InvestmentAccount(
                label="UK ETFs",
                market_value=512_000.0,
                cost_basis=201_000.0,
                asset_class="equity",
            )
        )

    accounts.append(
        InvestmentAccount(
            label="AU ETFs",
            market_value=63_000.0,
            cost_basis=63_000.0,
            asset_class="equity",
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
