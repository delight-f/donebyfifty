"""Entry point for the Monte Carlo Retirement Simulator.

Menu-driven interactive flow: new simulation, load profile, manage profiles.
"""

from __future__ import annotations

import dataclasses
import random as _random
import sys
from typing import NoReturn

from config import clear_tax_cache
from models import Household, Profile, ResultsSession, SimulationInputs
from rich.prompt import IntPrompt, Prompt
from simulation import run_monte_carlo
from ui import (
    _prompt_int,
    configure_household,
    configure_simulation_params,
    console,
    display_profile_list,
    display_results,
    edit_household,
    print_banner,
    review_before_run,
    show_results_menu,
)

from profiles import (
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)

# Module-level tracking for seed re-run
_last_household: Household | None = None
_last_inputs: SimulationInputs | None = None


def _new_simulation() -> None:
    """Run a new simulation from scratch."""
    from ui import choose_preset

    preset = choose_preset()

    # Prompt for start age BEFORE household config so earner super warnings
    # project caps and thresholds from the correct baseline age
    console.print("\n[bold bright_green]--- Simulation Start Age ---[/]")
    start_age = _prompt_int(
        "  Simulation start age",
        37,
        lo=18,
        hi=65,
    )

    household = configure_household(preset, start_age=start_age)
    sim_inputs = configure_simulation_params(household=household, start_age=start_age)

    if not review_before_run(household, sim_inputs):
        console.print("[yellow]Simulation cancelled.[/]")
        return

    _run_and_save(household, sim_inputs)


def _load_and_run() -> None:
    """Load a profile, optionally tweak, run simulation."""
    profiles = list_profiles()
    if not profiles:
        console.print("[yellow]No saved profiles found.[/]")
        return

    display_profile_list(profiles)

    choice = IntPrompt.ask(
        "[bright_green]Select profile number[/]",
        default=1,
        choices=[str(i) for i in range(1, len(profiles) + 1)],
    )
    selected = profiles[choice - 1]

    try:
        profile = load_profile(selected["name"])
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading profile: {e}[/]")
        return

    console.print(f"\n[bold]Loaded profile: {profile.profile_name}[/]")

    if profile.inputs.household.num_earners > 0:
        console.print(
            f"  {profile.inputs.household.num_earners} earner(s), "
            f"{profile.inputs.household.num_children} child(ren), "
            f"{profile.inputs.household.num_mortgages} mortgage(s)"
        )

    if profile.last_results:
        console.print(
            f"  Last run: {profile.last_results.p_success * 100:.1f}% success, "
            f"{profile.last_results.trials} trials"
        )

    start_age = profile.inputs.simulation_start_age

    tweak_profile = Prompt.ask(
        "[bright_green]Tweak profile inputs before running?[/] [y/N]",
        default="n",
    )
    tweak_profile_yn = tweak_profile.strip().lower() in ("y", "yes", "true")

    if tweak_profile_yn:
        household = edit_household(profile.inputs.household, start_age=start_age)
    else:
        household = profile.inputs.household

    # Always offer to tweak simulation parameters
    tune_sim = Prompt.ask(
        "[bright_green]Tweak simulation parameters before running?"
        " (iterations, inflation, etc.)[/] [y/N]",
        default="n",
    )
    if tune_sim.strip().lower() in ("y", "yes", "true"):
        sim_inputs = configure_simulation_params(
            profile.inputs, household=household, start_age=start_age
        )
    elif tweak_profile_yn:
        # Household tweaked, sim params not — must rebuild with new household
        sim_inputs = dataclasses.replace(profile.inputs, household=household)
    else:
        sim_inputs = profile.inputs

    if not review_before_run(household, sim_inputs):
        console.print("[yellow]Simulation cancelled.[/]")
        return

    _run_and_save(household, sim_inputs, profile)


def _run_and_save(
    household: Household,
    sim_inputs: SimulationInputs,
    existing_profile: Profile | None = None,
) -> None:
    """Run the simulation and optionally save results to a profile."""
    clear_tax_cache()

    console.print(f"\n[green]Running {sim_inputs.n_iterations:,} trials...[/]")

    # Auto-generate a seed if none provided for reproducibility tracking
    if sim_inputs.seed is None:
        seed = _random.randint(0, 2**31 - 1)
        sim_inputs = dataclasses.replace(sim_inputs, seed=seed)
        console.print(f"[dim]Seed: {seed}[/]")

    try:
        results = run_monte_carlo(household, sim_inputs, seed=sim_inputs.seed)
    except Exception as e:
        console.print(f"[red]Simulation error: {e}[/]")
        return

    display_results(
        results,
        household,
        start_age=sim_inputs.simulation_start_age,
        success_threshold=sim_inputs.success_threshold,
    )

    # ── Results menu (drill-down views) ────────────────────────────
    session = ResultsSession(
        results=results,
        household=household,
        inputs=sim_inputs,
    )
    show_results_menu(session)

    save_it = Prompt.ask(
        "[bright_green]Save this profile?[/] [Y/n]",
        default="y",
    )
    if save_it.strip().lower() not in ("y", "yes", "true"):
        return

    if existing_profile:
        profile = existing_profile
        profile.inputs = sim_inputs
        profile.last_results = results
    else:
        name = Prompt.ask("[bright_green]Profile name[/]", default="My Scenario")
        profile = Profile(
            profile_name=name,
            inputs=sim_inputs,
            last_results=results,
        )

    try:
        path = save_profile(profile)
        console.print(f"[green]Saved to {path}[/]")
    except OSError as e:
        console.print(f"[red]Could not save: {e}[/]")

    # Track for seed re-run
    global _last_household, _last_inputs
    _last_household = household
    _last_inputs = sim_inputs


def _run_with_seed() -> None:
    """Re-run the last simulation with a specific seed."""
    global _last_household, _last_inputs
    if _last_household is None or _last_inputs is None:
        console.print("[yellow]No previous simulation to re-run.[/]")
        return

    seed_raw = Prompt.ask(
        "[bright_green]Enter seed[/] (integer, or blank for random)",
        default="",
    )
    if seed_raw.strip():
        try:
            seed = int(seed_raw.strip())
        except ValueError:
            console.print("[red]Invalid seed — must be an integer.[/]")
            return
    else:
        seed = None

    inputs = dataclasses.replace(_last_inputs, seed=seed)
    _run_and_save(_last_household, inputs)


def _manage_profiles() -> None:
    """List, view, and delete saved profiles."""
    profiles = list_profiles()
    display_profile_list(profiles)

    if not profiles:
        return

    action = Prompt.ask(
        "[bright_green]Action[/]",
        choices=["delete", "back"],
        default="back",
    )

    if action == "delete":
        choice = IntPrompt.ask(
            "[bright_green]Number to delete[/]",
            choices=[str(i) for i in range(1, len(profiles) + 1)],
        )
        selected = profiles[choice - 1]
        confirm = Prompt.ask(
            f"[yellow]Delete '{selected['name']}'?[/] [y/N]",
            default="n",
        )
        if confirm.strip().lower() in ("y", "yes"):
            if delete_profile(selected["name"]):
                console.print(f"[green]Deleted '{selected['name']}'[/]")
            else:
                console.print(f"[red]Could not delete '{selected['name']}'[/]")


def main() -> NoReturn:
    """Banner then interactive menu loop."""
    print_banner()

    while True:
        console.print("\n[bold cyan]Main Menu[/]")
        console.print("  1. New simulation")
        console.print("  2. Load profile")
        console.print("  3. Manage profiles")
        console.print("  4. Exit")
        console.print("  5. Re-run last simulation with specific seed")

        choice = IntPrompt.ask(
            "[bright_green]Choice[/]",
            choices=["1", "2", "3", "4", "5"],
        )

        if choice == 1:
            _new_simulation()
        elif choice == 2:
            _load_and_run()
        elif choice == 3:
            _manage_profiles()
        elif choice == 4:
            console.print("[green]Goodbye.[/]")
            sys.exit(0)
        elif choice == 5:
            _run_with_seed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/]")
        sys.exit(1)
