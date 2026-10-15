"""Profile save/load/list/delete operations.

All file I/O lives here. Each profile is one JSON file in a per-user data
directory (see ``_profiles_dir``). Profiles are addressed by file name, as
returned by ``list_profiles``; ``profile_filename`` derives the file name
from a profile name.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Profile, ProfileError

# =============================================================================
# PATH RESOLUTION
# =============================================================================


def _profiles_dir() -> Path:
    r"""Return the directory profiles are stored in.  The directory is not created.

    Linux and macOS: ``$XDG_DATA_HOME/donebyfifty/profiles`` (default
    ``~/.local/share``).  Windows: ``%APPDATA%\donebyfifty\profiles``.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "donebyfifty" / "profiles"


def _sanitise_filename(name: str) -> str:
    """Strip unsafe characters from a profile name for use as a filename.

    Args:
        name: Raw profile name from user input.

    Returns:
        Safe filename (alphanumeric, hyphens, underscores, spaces only).
        Silently truncated to 120 characters to prevent filesystem limits
        being hit and to keep the many-to-one mapping from colliding
        wildly different names.

    """
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    safe = safe.strip() or "unnamed"
    return safe[:120]


def profile_filename(name: str) -> str:
    """Return the file name a profile called ``name`` is stored under."""
    return _sanitise_filename(name) + ".json"


def profile_exists(name: str) -> bool:
    """Return True if a profile file already exists for ``name``.

    Different names can sanitise to the same file name, so callers should
    confirm with the user before overwriting.
    """
    return (_profiles_dir() / profile_filename(name)).is_file()


# =============================================================================
# CRUD OPERATIONS
# =============================================================================


def save_profile(profile: Profile) -> Path:
    """Save a profile to disk as a JSON file.

    Write is atomic: data is flushed to a temporary file in the same
    directory, ``fsync`` ed, then ``os.replace`` ed over the target.  A crash
    or error mid-write leaves the previous profile intact.  Errors are not
    retried through another path; ``OSError`` propagates to the caller.

    Args:
        profile: The profile to save. Its ``updated_at`` is set to now.

    Returns:
        The path of the saved file.

    Raises:
        OSError: If the profiles directory cannot be written to.
        ValueError: If the profile contains NaN or infinity.

    """
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    if not profile.created_at:
        profile.created_at = profile.updated_at

    directory = _profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / profile_filename(profile.profile_name)
    payload = json.dumps(profile.to_dict(), indent=2, allow_nan=False)

    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".profile-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def load_profile(file: str | Path) -> Profile:
    """Load a profile from a file name as returned by ``list_profiles``.

    Any directory part of ``file`` is ignored, so the path cannot escape the
    profiles directory.

    Args:
        file: The profile's file name (e.g. ``"My plan.json"``).

    Returns:
        The deserialised ``Profile``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ProfileError: If the file is not valid JSON, is not a JSON object, or
            cannot be deserialised (including unsupported schema versions).

    """
    path = _profiles_dir() / Path(file).name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise ProfileError(f"Profile file '{path.name}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"Profile file '{path.name}' does not contain a JSON object.")
    return Profile.from_dict(data)


def list_profiles() -> list[dict[str, Any]]:
    """List all saved profiles with metadata.

    Every entry has the keys ``file``, ``name``, ``created_at``,
    ``updated_at``, ``num_earners``, ``num_children``, ``p_success`` (or None)
    and ``corrupt``.  Unreadable or invalid files are listed with
    ``corrupt=True`` and zeroed metadata so the user can still delete them.

    """
    results: list[dict[str, Any]] = []
    for path in sorted(_profiles_dir().glob("*.json")):
        try:
            profile = load_profile(path.name)
        except (OSError, ProfileError):
            results.append(
                {
                    "file": path.name,
                    "name": path.stem,
                    "created_at": "",
                    "updated_at": "",
                    "num_earners": 0,
                    "num_children": 0,
                    "p_success": None,
                    "corrupt": True,
                }
            )
            continue
        last_run = profile.last_results
        results.append(
            {
                "file": path.name,
                "name": profile.profile_name,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
                "num_earners": profile.inputs.household.num_earners,
                "num_children": profile.inputs.household.num_children,
                "p_success": last_run.p_success if last_run else None,
                "corrupt": False,
            }
        )
    return results


def delete_profile(file: str | Path) -> bool:
    """Delete a profile by file name as returned by ``list_profiles``.

    Returns:
        True if deleted, False if no such file exists.

    """
    path = _profiles_dir() / Path(file).name
    if not path.is_file():
        return False
    path.unlink()
    return True
