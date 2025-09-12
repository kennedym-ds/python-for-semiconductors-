"""Environment bootstrap utility for tiered requirements.

Usage examples (PowerShell):
  python env_setup.py --tier basic
  python env_setup.py --tier intermediate
  python env_setup.py --tier advanced
  python env_setup.py --tier full --force
  python env_setup.py --tier full --python 3.11

Features:
  * Creates a local .venv if missing (or recreates with --force)
  * Installs appropriate tier requirements (tier files chain upward)
  * Upgrades pip/setuptools/wheel for reliability
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
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"
VALID_TIERS = ["basic", "intermediate", "advanced", "full"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for environment setup.

    Returns:
        argparse.Namespace: Parsed arguments including tier, python version,
            force flag, and whether to skip installation.
    """
    parser = argparse.ArgumentParser(
        description="Create and install a tiered virtual environment"
    )
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
    return parser.parse_args()


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
    raise RuntimeError(
        f"Could not find a working python for version {version}. Ensure it is installed and on PATH."
    )


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


def install_requirements(python_path: Path, tier: str) -> None:
    """Install the requirements corresponding to a given tier.

    Args:
        python_path: Path to venv python.
        tier: Chosen tier name.
    """
    req_file = PROJECT_ROOT / f"requirements-{tier}.txt"
    if not req_file.exists():
        raise FileNotFoundError(f"Missing requirements file: {req_file}")
    print(f"[env-setup] Installing tier '{tier}' requirements from {req_file.name}...")
    subprocess.check_call(
        [
            str(python_path),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(req_file),
        ]
    )


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


def main() -> None:
    args = parse_args()
    if args.tier not in VALID_TIERS:
        raise SystemExit(f"Invalid tier {args.tier}; choose from {VALID_TIERS}")

    python_exec = resolve_python_executable(args.python_version)
    venv_dir = Path(args.venv_dir).resolve()

    create_virtualenv(python_exec, venv_dir, args.force)
    py_in_venv = venv_python(venv_dir)

    upgrade_tooling(py_in_venv)

    if not args.upgrade_only:
        install_requirements(py_in_venv, args.tier)
    else:
        print("[env-setup] Skipping dependency installation (--upgrade-only).")

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
