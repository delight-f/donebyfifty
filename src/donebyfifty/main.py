"""Entry point for the Monte Carlo Retirement Simulator.

Menu-driven interactive flow: new simulation, load profile, re-run with a seed,
manage profiles.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import random as _random
import traceback
from pathlib import Path

from rich.markup import escape
from rich.prompt import IntPrompt, Prompt

from .config import DEFAULT_SIM_START_AGE, THEME_COLOR_WARN
from .models import Household, Profile, ResultsSession, SimulationInputs
from .montecarlo import run_monte_carlo
from .profiles import (
    _profiles_dir,
    delete_profile,
    list_profiles,
    load_profile,
    profile_exists,
    save_profile,
)
from .ui_common import _prompt_int, _prompt_yn, console, display_profile_list, print_banner
from .ui_results import display_results, show_results_menu
from .ui_wizard import (
    choose_preset,
    configure_household,
    configure_simulation_params,
    edit_household,
    review_before_run,
)

# Most recent completed run, kept for "re-run with seed". Set as soon as the
# run finishes, so it is available even if the user does not save the profile.
_last_household: Household | None = None
_last_inputs: SimulationInputs | None = None


def _log_unexpected_error() -> Path:
    """Append the active exception's traceback to errors.log and return its path."""
    log_path = _profiles_dir().parent / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"--- {stamp} ---\n{traceback.format_exc()}\n")
    return log_path


def _review_and_run(
    household: Household,
    sim_inputs: SimulationInputs,
    existing_profile: Profile | None = None,
) -> None:
    """Show the review screen until the user runs the simulation or backs out."""
    while True:
        action = review_before_run(household, sim_inputs)
        if action == "run":
            _run_and_save(household, sim_inputs, existing_profile)
            return
        if action == "household":
            household = edit_household(household, start_age=sim_inputs.simulation_start_age)
            sim_inputs = dataclasses.replace(sim_inputs, household=household)
        elif action == "simulation":
            sim_inputs = configure_simulation_params(
                sim_inputs,
                household=household,
                start_age=sim_inputs.simulation_start_age,
            )
        else:
            console.print("[yellow]Simulation cancelled.[/]")
            return


def _new_simulation() -> None:
    """Run a new simulation from scratch."""
    preset = choose_preset()

    # Prompt for start age BEFORE household config so earner super warnings
    # project caps and thresholds from the correct baseline age
    console.print("\n[bold bright_green]--- Simulation Start Age ---[/]")
    start_age = _prompt_int(
        "  Simulation start age",
        DEFAULT_SIM_START_AGE,
        lo=18,
        hi=65,
    )

    household = configure_household(preset, start_age=start_age)
    sim_inputs = configure_simulation_params(household=household, start_age=start_age)
    _review_and_run(household, sim_inputs)


def _load_and_run() -> None:
    """Load a profile, optionally tweak, run simulation."""
    profiles = [p for p in list_profiles() if not p["corrupt"]]
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
        profile = load_profile(selected["file"])
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading profile: {escape(str(e))}[/]")
        return

    console.print(f"\n[bold]Loaded profile: {escape(profile.profile_name)}[/]")

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

    household = profile.inputs.household
    if _prompt_yn("Tweak profile inputs before running?", default=False):
        household = edit_household(household, start_age=start_age)

    if _prompt_yn(
        "Tweak simulation parameters before running? (iterations, inflation, etc.)",
        default=False,
    ):
        sim_inputs = configure_simulation_params(
            profile.inputs, household=household, start_age=start_age
        )
    else:
        # Keep the saved settings, but use the (possibly edited) household.
        sim_inputs = dataclasses.replace(profile.inputs, household=household)

    _review_and_run(household, sim_inputs, profile)


def _run_and_save(
    household: Household,
    sim_inputs: SimulationInputs,
    existing_profile: Profile | None = None,
) -> None:
    """Run the simulation and optionally save results to a profile."""
    global _last_household, _last_inputs

    console.print(f"\n[green]Running {sim_inputs.n_iterations:,} trials...[/]")

    # Auto-generate a seed if none provided for reproducibility tracking
    if sim_inputs.seed is None:
        seed = _random.randint(0, 2**31 - 1)
        sim_inputs = dataclasses.replace(sim_inputs, seed=seed)
        console.print(f"[dim]Seed: {seed}[/]")

    try:
        with console.status("[green]Simulating...[/]"):
            results = run_monte_carlo(
                dataclasses.replace(sim_inputs, household=household), seed=sim_inputs.seed
            )
    except Exception as e:
        log_path = _log_unexpected_error()
        console.print(f"[red]Simulation failed: {escape(str(e))}[/]")
        console.print(f"[dim]Details written to {escape(str(log_path))}[/]")
        return

    for message in results.warnings:
        console.print(f"[{THEME_COLOR_WARN}]Warning: {escape(message)}[/]")

    _last_household = household
    _last_inputs = sim_inputs

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

    if not _prompt_yn("Save this profile?", default=True):
        return

    if existing_profile:
        profile = existing_profile
        profile.inputs = sim_inputs
        profile.last_results = results
    else:
        name = Prompt.ask("[bright_green]Profile name[/]", default="My Scenario")
        if profile_exists(name) and not _prompt_yn(
            f"'{escape(name)}' already exists. Overwrite?", default=False
        ):
            return
        profile = Profile(
            profile_name=name,
            inputs=sim_inputs,
            last_results=results,
        )

    try:
        path = save_profile(profile)
        console.print(f"[green]Saved to {escape(str(path))}[/]")
    except OSError as e:
        console.print(f"[red]Could not save: {escape(str(e))}[/]")


def _run_with_seed() -> None:
    """Re-run the last simulation, with a new random seed or a chosen one."""
    if _last_household is None or _last_inputs is None:
        console.print("[yellow]No previous simulation to re-run.[/]")
        return

    if _prompt_yn("Use a new random seed?", default=True):
        seed = None
    else:
        seed = _prompt_int("  Seed (integer)", lo=0, hi=2**31 - 1)

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
        name = escape(selected["name"])
        if _prompt_yn(f"Delete '{name}'?", default=False):
            if delete_profile(selected["file"]):
                console.print(f"[green]Deleted '{name}'[/]")
            else:
                console.print(f"[red]Could not delete '{name}'[/]")


def main() -> None:
    """Banner then interactive menu loop. Returns when the user exits."""
    print_banner()

    while True:
        try:
            console.print("\n[bold cyan]Main Menu[/]")
            console.print("  1. New simulation")
            console.print("  2. Load profile")
            console.print("  3. Re-run last simulation with a seed")
            console.print("  4. Manage profiles")
            console.print("  5. Exit")

            choice = IntPrompt.ask(
                "[bright_green]Choice[/]",
                choices=["1", "2", "3", "4", "5"],
                default=5,
            )

            if choice == 5:
                console.print("[green]Goodbye.[/]")
                return
            if choice == 1:
                _new_simulation()
            elif choice == 2:
                _load_and_run()
            elif choice == 3:
                _run_with_seed()
            else:
                _manage_profiles()
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled. Back to main menu.[/]")
        except EOFError:
            console.print("\n[green]Goodbye.[/]")
            return
        except Exception as e:
            log_path = _log_unexpected_error()
            console.print(f"\n[red]Unexpected error: {escape(str(e))}[/]")
            console.print(f"[dim]Traceback written to {escape(str(log_path))}[/]")


if __name__ == "__main__":
    main()
