"""Tests for the profile management menu in ``main.py``."""

from __future__ import annotations

from typing import Any

import pytest

from donebyfifty import main


def _profile_entry() -> dict[str, Any]:
    return {"file": "gone.json", "name": "Gone", "created_at": "2026-01-01T00:00:00"}


def _choose(monkeypatch: pytest.MonkeyPatch, *, delete_result: bool) -> list[str]:
    """Script the menu: pick 'delete', choose profile 1, confirm, and record deletions."""
    deleted: list[str] = []

    def fake_delete(file: str) -> bool:
        deleted.append(file)
        return delete_result

    monkeypatch.setattr(main, "list_profiles", lambda: [_profile_entry()])
    monkeypatch.setattr(main, "display_profile_list", lambda profiles: None)
    monkeypatch.setattr(main.Prompt, "ask", lambda *args, **kwargs: "delete")
    monkeypatch.setattr(main.IntPrompt, "ask", lambda *args, **kwargs: 1)
    monkeypatch.setattr(main, "_prompt_yn", lambda *args, **kwargs: True)
    monkeypatch.setattr(main, "delete_profile", fake_delete)
    return deleted


def test_manage_profiles_deletes_selected_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deleted = _choose(monkeypatch, delete_result=True)

    main._manage_profiles()

    assert deleted == ["gone.json"]
    assert "Deleted 'Gone'" in capsys.readouterr().out


def test_manage_profiles_reports_delete_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deleted = _choose(monkeypatch, delete_result=False)

    main._manage_profiles()

    assert deleted == ["gone.json"]
    out = capsys.readouterr().out
    assert "Could not delete 'Gone'" in out
    assert "Deleted 'Gone'" not in out


def test_manage_profiles_without_profiles_returns_before_prompting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_prompt(*args: object, **kwargs: object) -> str:
        raise AssertionError("Prompt should not be shown when there are no profiles")

    monkeypatch.setattr(main, "list_profiles", lambda: [])
    monkeypatch.setattr(main, "display_profile_list", lambda profiles: None)
    monkeypatch.setattr(main.Prompt, "ask", no_prompt)

    main._manage_profiles()
