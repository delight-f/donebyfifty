"""Tests for profile persistence: atomic save, load, list, delete, and upgrades."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from donebyfifty import profiles
from donebyfifty.models import (
    PROFILE_VERSION,
    Earner,
    Household,
    InvestmentAccount,
    Profile,
    ProfileError,
    SimulationInputs,
)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the profile store at a fresh directory under tmp_path."""
    directory = tmp_path / "profiles"
    monkeypatch.setattr(profiles, "_profiles_dir", lambda: directory)
    return directory


def _make_profile(name: str = "Plan", n_iterations: int = 1_000) -> Profile:
    household = Household(earners=(Earner(label="Me", super_access_age=60),))
    return Profile(
        profile_name=name,
        inputs=SimulationInputs(household=household, n_iterations=n_iterations),
    )


def _write_json(directory: Path, filename: str, data: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestSaveAndLoad:
    def test_roundtrip(self, store: Path) -> None:
        path = profiles.save_profile(_make_profile("My plan", n_iterations=2_500))
        assert path.parent == store
        assert path.name == profiles.profile_filename("My plan")

        loaded = profiles.load_profile(path.name)
        assert loaded.profile_name == "My plan"
        assert loaded.inputs.n_iterations == 2_500

    def test_save_sets_timestamps(self, store: Path) -> None:
        profile = _make_profile()
        profiles.save_profile(profile)
        assert profile.created_at
        assert profile.updated_at

    def test_load_ignores_directory_part(self, store: Path) -> None:
        path = profiles.save_profile(_make_profile("Plan"))
        loaded = profiles.load_profile(f"../elsewhere/{path.name}")
        assert loaded.profile_name == "Plan"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
    def test_saved_file_is_private(self, store: Path) -> None:
        path = profiles.save_profile(_make_profile())
        assert path.stat().st_mode & 0o777 == 0o600

    def test_nan_is_rejected_and_nothing_written(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Profile, "to_dict", lambda self: {"x": float("nan")})
        with pytest.raises(ValueError):
            profiles.save_profile(_make_profile("Bad"))
        assert list(store.glob("*")) == []

    def test_failed_replace_keeps_original_and_removes_temp(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profiles.save_profile(_make_profile("Plan", n_iterations=1_000))
        real_replace = profiles.os.replace

        def boom(src: str, dst: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(profiles.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            profiles.save_profile(_make_profile("Plan", n_iterations=9_999))

        monkeypatch.setattr(profiles.os, "replace", real_replace)
        assert profiles.load_profile(profiles.profile_filename("Plan")).inputs.n_iterations == 1_000
        assert not [p for p in store.iterdir() if p.name.startswith(".profile-")]


class TestListAndDelete:
    def test_list_reports_metadata(self, store: Path) -> None:
        profiles.save_profile(_make_profile("Beta"))
        profiles.save_profile(_make_profile("Alpha"))

        entries = profiles.list_profiles()
        assert [e["name"] for e in entries] == ["Alpha", "Beta"]
        first = entries[0]
        assert first["corrupt"] is False
        assert first["num_earners"] == 1
        assert first["num_children"] == 0
        assert first["p_success"] is None

    def test_corrupt_file_is_listed_but_not_loadable(self, store: Path) -> None:
        store.mkdir(parents=True, exist_ok=True)
        (store / "broken.json").write_text("{not json", encoding="utf-8")

        entries = profiles.list_profiles()
        assert len(entries) == 1
        assert entries[0]["file"] == "broken.json"
        assert entries[0]["corrupt"] is True

        with pytest.raises(ProfileError):
            profiles.load_profile("broken.json")

    def test_non_object_json_is_rejected(self, store: Path) -> None:
        _write_json(store, "list.json", [1, 2, 3])
        with pytest.raises(ProfileError, match="JSON object"):
            profiles.load_profile("list.json")

    def test_unsupported_version_is_rejected(self, store: Path) -> None:
        _write_json(store, "future.json", {"_version": 9, "profile_name": "Future"})
        with pytest.raises(ProfileError, match="version 9"):
            profiles.load_profile("future.json")

    def test_missing_required_field_is_profile_error(self, store: Path) -> None:
        data = {
            "_version": 3,
            "profile_name": "Incomplete",
            "inputs": {"household": {"earners": [{"label": "Y"}]}},
        }
        _write_json(store, "incomplete.json", data)
        with pytest.raises(ProfileError):
            profiles.load_profile("incomplete.json")

    def test_delete(self, store: Path) -> None:
        path = profiles.save_profile(_make_profile("Doomed"))
        assert profiles.delete_profile(path.name) is True
        assert not path.exists()
        assert profiles.delete_profile(path.name) is False

    def test_delete_corrupt_file(self, store: Path) -> None:
        store.mkdir(parents=True, exist_ok=True)
        (store / "broken.json").write_text("{", encoding="utf-8")
        assert profiles.delete_profile("broken.json") is True
        assert profiles.list_profiles() == []

    def test_profile_exists(self, store: Path) -> None:
        assert profiles.profile_exists("Plan") is False
        profiles.save_profile(_make_profile("Plan"))
        assert profiles.profile_exists("Plan") is True


class TestUpgrade:
    def test_v2_file_loads_as_v3(self, store: Path) -> None:
        """A v2 file in the current serialiser layout upgrades on load."""
        household = Household(
            earners=(Earner(super_access_age=60, pt_days_per_week=1.0),),
            investment_accounts=(InvestmentAccount(label="Fixed", interest_rate=0.0),),
        )
        data = Profile(
            profile_name="Legacy", inputs=SimulationInputs(household=household)
        ).to_dict()
        data["_version"] = 2
        data["inputs"]["household"]["earners"][0]["pt_start_age"] = -1
        data["inputs"]["household"]["earners"][0]["pt_end_age"] = 0
        _write_json(store, "legacy.json", data)

        loaded = profiles.load_profile("legacy.json")
        earner = loaded.inputs.household.earners[0]
        assert earner.pt_start_age is None
        assert earner.pt_end_age is None
        assert loaded.inputs.household.investment_accounts[0].interest_rate is None

        # Re-saving writes the current schema version.
        resaved = json.loads(profiles.save_profile(loaded).read_text(encoding="utf-8"))
        assert resaved["_version"] == PROFILE_VERSION


class TestDirectory:
    @pytest.mark.skipif(os.name == "nt", reason="POSIX data directory")
    def test_xdg_data_home_is_used_and_not_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data_home = tmp_path / "data"
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
        resolved = profiles._profiles_dir()
        assert resolved == data_home / "donebyfifty" / "profiles"
        assert not data_home.exists()
