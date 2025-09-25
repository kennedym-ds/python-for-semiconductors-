"""Environment bootstrap utility for tiered requirements.

Usage examples (PowerShell):
  python env_setup.py --tier basic
  python env_setup.py --tier intermediate
  python env_setup.py --tier advanced
  python env_setup.py --tier full --force
  python env_setup.py --tier full --python 3.11

Features:
    * Creates a local .venv if missing (or recreates with --force)
    * Installs tiered lockfiles (`requirements-<tier>.txt`) for deterministic environments
    * Optionally installs via pyproject dependency groups when requested and supported
    * Supports additional dependency groups (e.g., docs, dev) via --extras, with lockfile fallback when available
    * Upgrades pip/setuptools/wheel for reliability (pip >= 25.1 enables --group)
    * Provides clear console guidance for activation on Windows / Unix

Tiers hierarchy:
  basic -> intermediate -> advanced -> full

Design notes:
  We intentionally keep this script standard-library only so it functions
  before any dependencies are installed. Error messages are explicit so
  newcomers can troubleshoot path / Python version issues quickly.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback handled later
    tomllib = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).parent
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"
VALID_TIERS = ["basic", "intermediate", "advanced", "full"]
# pip install --group became available in pip 25.1
# Reference: https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-pip-install-group
MIN_GROUP_VERSION = (25, 1, 0)
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
KNOWN_GROUP_FALLBACK = {
    "basic",
    "intermediate",
    "advanced",
    "full",
    "docs",
    "dev",
}


def discover_dependency_groups() -> set[str]:
    """Return dependency group names declared in pyproject.toml (best effort)."""

    groups = set(KNOWN_GROUP_FALLBACK)
    if tomllib is None or not PYPROJECT_PATH.exists():
        return groups
    try:
        with PYPROJECT_PATH.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception:  # pragma: no cover - safe fallback
        return groups
    declared = data.get("dependency-groups", {})
    if isinstance(declared, dict):
        groups.update(str(name) for name in declared.keys())
    return groups


KNOWN_DEPENDENCY_GROUPS = discover_dependency_groups()
VALID_EXTRA_GROUPS = sorted(set(KNOWN_DEPENDENCY_GROUPS) - set(VALID_TIERS))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for environment setup.

    Returns:
        argparse.Namespace: Parsed arguments including tier, python version,
            force flag, and whether to skip installation.
    """
    parser = argparse.ArgumentParser(description="Create and install a tiered virtual environment")
    parser.add_argument(
        "--tier",
        required=True,
        choices=VALID_TIERS,
        help="Dependency tier to install (basic|intermediate|advanced|full)",
    )
    parser.add_argument(
        "--python",
        dest="python_version",
        help="Optional target Python version (e.g., 3.11). Must be available on PATH.",
    )
    parser.add_argument(
        "--venv-dir",
        default=str(DEFAULT_VENV_DIR),
        help="Virtual environment directory (default: ./.venv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the virtual environment if it already exists.",
    )
    parser.add_argument(
        "--upgrade-only",
        action="store_true",
        help="Skip dependency install; only ensure pip tooling upgraded.",
    )
    parser.add_argument(
        "--use-groups",
        action="store_true",
        help="Install via pip dependency groups instead of lockfiles (requires pip >= 25.1).",
    )
    extras_help = "Additional dependency groups (e.g., docs, dev) to install on top of the tier."
    if VALID_EXTRA_GROUPS:
        extras_help += f" Available groups: {', '.join(VALID_EXTRA_GROUPS)}."
    parser.add_argument(
        "--extras",
        nargs="+",
        default=[],
        help=extras_help,
    )
    return parser.parse_args()


def ordered_unique(items: list[str]) -> list[str]:
    """Return items with natural order preserved and duplicates removed."""

    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def normalize_extras(extras: list[str], tier: str) -> list[str]:
    """Validate requested extras and ensure they are compatible with the tier."""

    if not extras:
        return []
    normalized = ordered_unique([extra.strip() for extra in extras if extra.strip()])
    if not normalized:
        return []
    tier_set = set(VALID_TIERS)
    unknown = [extra for extra in normalized if extra not in KNOWN_DEPENDENCY_GROUPS]
    if unknown:
        known_display = ", ".join(sorted(KNOWN_DEPENDENCY_GROUPS))
        raise SystemExit(f"Unknown dependency group(s): {', '.join(unknown)}. Known groups: {known_display}.")
    conflicting = [extra for extra in normalized if extra in tier_set]
    if conflicting:
        raise SystemExit("Extras may not reference tier names directly. " f"Received: {', '.join(conflicting)}.")
    if tier in normalized:
        raise SystemExit("Extras list must not include the selected tier.")
    return normalized


def resolve_python_executable(version: str | None) -> str:
    """Resolve the python executable according to optional version.

    We attempt pythonX.Y then fallback to 'python'.

    Args:
        version: Optional version string like '3.11'.

    Returns:
        The executable name/path to use.
    """
    if not version:
        return sys.executable
    candidates = [f"python{version}", sys.executable]
    for cand in candidates:
        try:
            out = subprocess.check_output([cand, "--version"], stderr=subprocess.STDOUT)
            if out:
                return cand
        except Exception:
            continue
    raise RuntimeError(f"Could not find a working python for version {version}. Ensure it is installed and on PATH.")


def create_virtualenv(python_exec: str, venv_dir: Path, force: bool) -> None:
    """Create (or recreate) the virtual environment.

    Args:
        python_exec: Python executable path/name to use.
        venv_dir: Target virtual environment directory.
        force: If True, delete existing directory first.
    """
    if venv_dir.exists() and force:
        print(f"[env-setup] Removing existing virtual environment: {venv_dir}")
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        print(f"[env-setup] Creating virtual environment at {venv_dir}")
        subprocess.check_call([python_exec, "-m", "venv", str(venv_dir)])
    else:
        print(f"[env-setup] Virtual environment already exists: {venv_dir}")


def venv_python(venv_dir: Path) -> Path:
    """Return path to the python executable inside the virtual environment."""
    if os.name == "nt":  # Windows
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def upgrade_tooling(python_path: Path) -> None:
    """Upgrade pip, setuptools, and wheel for reliable installs."""
    print("[env-setup] Upgrading packaging tooling (pip, setuptools, wheel)...")
    subprocess.check_call(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )


def parse_pip_version(output: str) -> tuple[int, int, int]:
    """Extract the pip version triple from ``pip --version`` output."""
    match = re.search(r"pip (\d+)\.(\d+)(?:\.(\d+))?", output)
    if not match:
        return (0, 0, 0)
    major, minor, patch = match.groups(default="0")
    return int(major), int(minor), int(patch)


def pip_supports_dependency_groups(python_path: Path) -> bool:
    """Return True when the environment's pip understands ``--group`` installs."""
    try:
        output = subprocess.check_output(
            [str(python_path), "-m", "pip", "--version"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    version = parse_pip_version(output)
    return version >= MIN_GROUP_VERSION


def build_lockfile_name(tier: str, extras: list[str]) -> str:
    """Return the canonical lockfile name for a tier and optional extras."""

    if extras:
        suffix = "+".join([tier, *extras])
    else:
        suffix = tier
    return f"requirements-{suffix}.txt"


def install_dependencies(
    python_path: Path,
    tier: str,
    extras: list[str],
    use_groups: bool,
    pip_groups_supported: bool | None = None,
) -> bool:
    """Install dependencies for the requested tier and extras.

    Args:
        python_path: Interpreter inside the virtual environment.
        tier: Primary dependency group to install.
        extras: Normalized extra dependency groups.
        use_groups: Whether to use pip dependency groups for installation.
        pip_groups_supported: Optional cached capability flag.

    Returns:
        bool: True when extras still need installation (handled via dependency groups).
    """

    extras = list(extras)
    if pip_groups_supported is None:
        pip_groups_supported = pip_supports_dependency_groups(python_path)

    if use_groups:
        if not pip_groups_supported:
            raise RuntimeError(
                "pip >= 25.1 is required for dependency group installs. Upgrade pip or rerun without --use-groups."
            )
        print(f"[env-setup] Installing tier '{tier}' via dependency group (pyproject.toml)...")
        subprocess.check_call(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--group",
                tier,
            ],
            cwd=str(PROJECT_ROOT),
        )
        return bool(extras)

    lock_name = build_lockfile_name(tier, extras)
    lock_file = PROJECT_ROOT / lock_name

    if not lock_file.exists():
        if extras:
            extras_label = ", ".join(extras)
            raise FileNotFoundError(
                f"No combined lockfile found for tier '{tier}' with extras [{extras_label}]. "
                "Generate it with tools/compile_tier_lockfiles.py --extras ..."
            )
        raise FileNotFoundError(
            f"No lockfile found for tier '{tier}'. Generate it with tools/compile_tier_lockfiles.py."
        )

    descriptor = tier if not extras else f"{tier} + {' + '.join(extras)}"
    print(f"[env-setup] Installing {descriptor} from lockfile {lock_file.name}...")
    subprocess.check_call(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(lock_file),
        ]
    )
    return False


def summarize_activation(venv_dir: Path) -> None:
    """Print activation instructions for different shells."""
    if os.name == "nt":
        win_ps = venv_dir / "Scripts" / "Activate.ps1"
        win_cmd = venv_dir / "Scripts" / "activate.bat"
        print("\nActivation instructions:")
        print(f"  PowerShell:  . {win_ps}")
        print(f"  Cmd.exe:     {win_cmd}")
    else:
        print("\nActivation instructions:")
        print(f"  Bash/Zsh: source {venv_dir}/bin/activate")
    print('\nTo verify: python -c "import sys; print(sys.executable)"')


def install_extra_groups(python_path: Path, extras: list[str], pip_groups_supported: bool) -> None:
    """Install additional dependency groups after the base tier."""

    if not extras:
        return
    if not pip_groups_supported:
        raise RuntimeError(
            "Additional dependency groups require pip >= 25.1 for --group installs. "
            "Rerun after upgrading pip or omit --extras."
        )
    for group in extras:
        print(f"[env-setup] Installing extra dependency group '{group}'...")
        subprocess.check_call(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--group",
                group,
            ],
            cwd=str(PROJECT_ROOT),
        )


def main() -> None:
    args = parse_args()
    if args.tier not in VALID_TIERS:
        raise SystemExit(f"Invalid tier {args.tier}; choose from {VALID_TIERS}")
    extras = normalize_extras(args.extras, args.tier)

    python_exec = resolve_python_executable(args.python_version)
    venv_dir = Path(args.venv_dir).resolve()

    create_virtualenv(python_exec, venv_dir, args.force)
    py_in_venv = venv_python(venv_dir)

    upgrade_tooling(py_in_venv)
    pip_groups_supported = pip_supports_dependency_groups(py_in_venv)

    effective_use_groups = args.use_groups
    combined_lockfile = PROJECT_ROOT / build_lockfile_name(args.tier, extras)

    if extras and not effective_use_groups:
        if combined_lockfile.exists():
            pass
        elif pip_groups_supported:
            print(
                f"[env-setup] Combined lockfile {combined_lockfile.name} not found; "
                "falling back to dependency group installs for extras."
            )
            effective_use_groups = True
        else:
            raise FileNotFoundError(
                f"Combined lockfile {combined_lockfile.name} not found and pip lacks --group support. "
                "Generate the lockfile or upgrade pip."
            )

    if not args.upgrade_only:
        extras_pending = install_dependencies(
            py_in_venv,
            args.tier,
            extras,
            use_groups=effective_use_groups,
            pip_groups_supported=pip_groups_supported,
        )
        if extras_pending:
            install_extra_groups(py_in_venv, extras, pip_groups_supported)
    else:
        print("[env-setup] Skipping dependency installation (--upgrade-only).")
        if extras:
            print("[env-setup] --extras is ignored when using --upgrade-only.")

    summarize_activation(venv_dir)
    print("\n[env-setup] Completed successfully.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted by user.")
    except Exception as exc:  # Provide clear error channel
        print(f"[env-setup] ERROR: {exc}")
        sys.exit(1)
