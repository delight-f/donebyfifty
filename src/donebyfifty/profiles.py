"""Profile save/load/list/delete operations.

All file I/O lives here. Uses JSON files, one per profile, stored in a
``profiles/`` directory relative to the executable or script location.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import Profile

# =============================================================================
# PATH RESOLUTION (handles PyInstaller-bundled executables)
# =============================================================================


def _profiles_dir() -> Path:
    """Resolve the profiles directory path — always the folder
    containing the actual exe on disk, regardless of build tool
    or how/where the exe was launched from.
    """  # noqa: D205
    if "__compiled__" in globals():
        # Nuitka (onefile or standalone): sys.argv[0] is the real exe path,
        # NOT the temp extraction dir. This is reliable in both modes.
        base = Path(sys.argv[0]).resolve().parent
    elif getattr(sys, "frozen", False):
        # PyInstaller
        base = Path(sys.executable).resolve().parent
    else:
        # Plain python main.py
        base = Path(__file__).resolve().parent

    profiles_path = base / "profiles"
    profiles_path.mkdir(exist_ok=True)
    return profiles_path


def _sanitise_filename(name: str) -> str:
    """Strip unsafe characters from a profile name for use as a filename.

    Args:
        name: Raw profile name from user input.

    Returns:
        Safe filename (alphanumeric, hyphens, underscores, spaces only).

    """
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    return safe.strip() or "unnamed"


# =============================================================================
# CRUD OPERATIONS
# =============================================================================


def save_profile(profile: Profile) -> Path:
    """Save a profile to disk as a JSON file.

    Args:
        profile: The profile to save. Its ``updated_at`` is set to now.

    Returns:
        The path of the saved file.

    Raises:
        OSError: If the profiles directory cannot be written to.

    """
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    if not profile.created_at:
        profile.created_at = profile.updated_at

    filename = _sanitise_filename(profile.profile_name) + ".json"
    path = _profiles_dir() / filename
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def load_profile(profile_name: str) -> Profile:
    """Load a single profile by name (without ``.json`` extension).

    Args:
        profile_name: The profile name (filename stem).

    Returns:
        The deserialised ``Profile``.

    Raises:
        FileNotFoundError: If no profile with that name exists.
        json.JSONDecodeError: If the profile file is corrupt.

    """
    filename = _sanitise_filename(profile_name) + ".json"
    path = _profiles_dir() / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    return Profile.from_dict(data)


def list_profiles() -> list[dict[str, Any]]:
    """List all saved profiles with metadata.

    Returns:
        A list of dicts with keys: ``name``, ``created_at``, ``updated_at``,
        ``num_earners``, ``num_children``, ``p_success`` (or None).

    """
    profiles_dir = _profiles_dir()
    results: list[dict[str, Any]] = []

    for fpath in sorted(profiles_dir.glob("*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            profile = Profile.from_dict(data)
            last_run = profile.last_results
            results.append(
                {
                    "name": profile.profile_name,
                    "created_at": profile.created_at,
                    "updated_at": profile.updated_at,
                    "num_earners": profile.inputs.household.num_earners,
                    "num_children": profile.inputs.household.num_children,
                    "p_success": last_run.p_success if last_run else None,
                    "file": fpath.name,
                }
            )
        except Exception:
            # Skip corrupt or unparseable files
            results.append(
                {
                    "name": fpath.stem,
                    "created_at": "",
                    "updated_at": "",
                    "num_earners": 0,
                    "num_children": 0,
                    "p_success": None,
                    "file": fpath.name,
                    "corrupt": True,
                }
            )

    return results


def delete_profile(profile_name: str) -> bool:
    """Delete a profile by name.

    Args:
        profile_name: The profile name (filename stem).

    Returns:
        True if deleted, False if not found.

    """
    filename = _sanitise_filename(profile_name) + ".json"
    path = _profiles_dir() / filename
    if path.exists():
        path.unlink()
        return True
    return False
