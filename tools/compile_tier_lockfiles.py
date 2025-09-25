"""Utility for generating pinned tier lockfiles with pip-tools.

This script flattens the tier dependency groups declared in ``pyproject.toml``
and feeds them to ``pip-compile`` so that the canonical ``requirements-<tier>.txt``
files remain in sync with the single source of truth. When ``--extras`` are
supplied it also emits combined lockfiles (for example
``requirements-advanced+docs.txt``) so that environments without ``pip
install --group`` support can still install tier + extra blends deterministically.
It intentionally avoids any third-party imports beyond ``pip-tools`` so that it
can run inside the existing development environment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - informative failure
        raise SystemExit("Unable to import tomllib/tomli. Run with Python 3.11+ or install 'tomli'.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
DEFAULT_TIERS = ("basic", "intermediate", "advanced", "full")


def load_pyproject() -> dict:
    """Parse ``pyproject.toml`` into a nested dictionary."""
    if not PYPROJECT_PATH.exists():  # pragma: no cover - repository invariant
        raise SystemExit(f"pyproject.toml not found at {PYPROJECT_PATH}")
    with PYPROJECT_PATH.open("rb") as fh:
        return tomllib.load(fh)


def ensure_piptools_available() -> None:
    """Provide a clear error if pip-tools is missing."""
    try:  # Delayed import for friendlier message
        import piptools  # type: ignore[unused-import]
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "pip-tools is required to compile lockfiles. Install it with 'python -m pip install pip-tools'."
        ) from exc


def ordered_dedupe(items: Iterable[str]) -> list[str]:
    """Return items with duplicates removed while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def flatten_group(name: str, groups: dict[str, list], stack: list[str] | None = None) -> list[str]:
    """Recursively expand ``include-group`` references for a dependency group."""
    if stack is None:
        stack = []
    if name in stack:
        cycle = " -> ".join(stack + [name])
        raise ValueError(f"Detected circular dependency group inclusion: {cycle}")
    if name not in groups:
        raise KeyError(f"Dependency group '{name}' is not defined in pyproject.toml")

    stack.append(name)
    expanded: list[str] = []
    for entry in groups[name]:
        if isinstance(entry, dict):
            include = entry.get("include-group")
            if not include:
                raise ValueError(f"Unsupported mapping in dependency group '{name}': {entry}")
            expanded.extend(flatten_group(include, groups, stack))
        else:
            expanded.append(str(entry))
    stack.pop()
    return expanded


def build_requirement_list(groups: Iterable[str], project_data: dict) -> list[str]:
    """Assemble the full requirement list for one or more dependency groups."""

    project_meta = project_data.get("project", {})
    base_requirements = project_meta.get("dependencies", [])
    dependency_groups: dict[str, list] = project_data.get("dependency-groups", {})

    combined = list(base_requirements)
    for group in groups:
        if group not in dependency_groups:
            raise KeyError(f"Dependency group '{group}' is not defined in pyproject.toml")
        combined.extend(flatten_group(group, dependency_groups))

    return ordered_dedupe(str(req) for req in combined)


def normalize_extra_groups(extras: Iterable[str], available: Iterable[str]) -> list[str]:
    """Validate and normalize extras to ensure they exist in pyproject.toml."""

    normalized = ordered_dedupe(str(extra).strip() for extra in extras if str(extra).strip())
    if not normalized:
        return []

    available_set = set(available)
    missing = [extra for extra in normalized if extra not in available_set]
    if missing:
        raise SystemExit("Unknown dependency group(s) supplied via --extras: " + ", ".join(missing))

    return normalized


def iter_extra_combinations(extras: list[str]) -> Iterable[list[str]]:
    """Yield ordered non-empty combinations of extras preserving declaration order."""

    if not extras:
        return []
    for r in range(1, len(extras) + 1):
        for combo in combinations(extras, r):
            yield list(combo)


def build_output_filename(prefix: str, tier: str, extras: Iterable[str]) -> str:
    """Construct the output filename for a tier + extras combination."""

    extras_list = list(extras)
    if extras_list:
        suffix = "+".join([tier, *extras_list])
    else:
        suffix = tier
    return f"{prefix}-{suffix}.txt"


def run_pip_compile(source_lines: list[str], output_file: Path, upgrade: bool) -> None:
    """Invoke pip-compile with a temporary .in constructed from the source lines."""
    with NamedTemporaryFile("w", suffix=".in", delete=False) as tmp_in:
        tmp_in.write("\n".join(source_lines) + "\n")
        tmp_path = Path(tmp_in.name)

    cmd = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        str(tmp_path),
        "--output-file",
        str(output_file),
    ]
    if upgrade:
        cmd.append("--upgrade")

    print(f"[lock] Compiling {output_file.name} ...")
    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    finally:
        tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pinned tier lockfiles with pip-compile")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(DEFAULT_TIERS),
        help="Dependency groups to compile (default: basic intermediate advanced full)",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to pip-compile to refresh to the latest compatible versions.",
    )
    parser.add_argument(
        "--output-prefix",
        default="requirements",
        help="Filename prefix for generated lockfiles (default: requirements).",
    )
    parser.add_argument(
        "--extras",
        nargs="+",
        default=[],
        help=(
            "Optional dependency groups (e.g., docs, dev) to combine with each selected tier. "
            "All non-empty combinations will be compiled in addition to the base tier lockfile."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_piptools_available()
    data = load_pyproject()
    dependency_groups = data.get("dependency-groups", {})

    extras = normalize_extra_groups(args.extras, dependency_groups.keys())
    extra_combos = list(iter_extra_combinations(extras)) if extras else []

    if extras and not args.tiers:
        raise SystemExit("At least one --tier must be specified when --extras are provided.")

    for tier in args.tiers:
        requirements = build_requirement_list([tier], data)
        output_name = build_output_filename(args.output_prefix, tier, [])
        output_path = PROJECT_ROOT / output_name
        run_pip_compile(requirements, output_path, upgrade=args.upgrade)

        for combo in extra_combos:
            combo_groups = [tier, *combo]
            combo_requirements = build_requirement_list(combo_groups, data)
            combo_name = build_output_filename(args.output_prefix, tier, combo)
            combo_path = PROJECT_ROOT / combo_name
            run_pip_compile(combo_requirements, combo_path, upgrade=args.upgrade)

    print("[lock] Completed lockfile compilation.")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
