#!/usr/bin/env python3
"""
Verify dataset path usage across the repository.

This tool enforces the dataset path conventions defined in the project:
- Notebooks use relative paths from their module location
- Module 2/3 notebooks must set: DATA_DIR = Path('../../../datasets').resolve()
- All datasets organized in subfolders: datasets/<name>/<file>
- Never use flat paths like: datasets/secom.data

Usage (examples):
  python verify_dataset_paths.py --format json
  python verify_dataset_paths.py --paths modules docs --fail-on warning

Exit codes:
- 0: No issues found
- 2: Issues found (errors or warnings depending on --fail-on)

Output:
- JSON or text summary of findings per file, with line numbers and snippets

This script is safe to run in CI and locally. It does not modify files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Default constants
DEFAULT_INCLUDE_DIRS: Tuple[str, ...] = ("modules", "docs", "datasets")
DEFAULT_FILE_EXTS: Tuple[str, ...] = (".ipynb", ".py", ".md")
IGNORE_DIR_NAMES: Tuple[str, ...] = (
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "temp_models",
)


@dataclass
class Issue:
    file: str
    line: int
    kind: str  # 'error' | 'warning'
    rule: str
    message: str
    snippet: str


@dataclass
class Report:
    root: str
    scanned_files: int
    issues: List[Issue]

    def to_json(self) -> str:
        return json.dumps(
            {
                "root": self.root,
                "scanned_files": self.scanned_files,
                "issue_count": len(self.issues),
                "issues": [asdict(i) for i in self.issues],
            },
            indent=2,
        )


def iter_files(base: Path, include_dirs: Sequence[str], exts: Sequence[str]) -> Iterable[Path]:
    for rel_dir in include_dirs:
        start = base / rel_dir
        if not start.exists():
            continue
        for path in start.rglob("*"):
            if path.is_dir():
                if path.name in IGNORE_DIR_NAMES:
                    # Skip ignored directories
                    for _ in []:
                        pass
                continue
            if path.suffix.lower() in exts:
                # Skip ignored directory names appearing in parent parts
                if any(part in IGNORE_DIR_NAMES for part in path.parts):
                    continue
                yield path


def read_text_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        # As a fallback, return empty to avoid failing the entire run
        return []


def extract_notebook_sources(nb_path: Path) -> List[str]:
    try:
        import json as _json

        data = _json.loads(nb_path.read_text(encoding="utf-8", errors="ignore"))
        cells = data.get("cells", [])
        lines: List[str] = []
        for c in cells:
            if c.get("cell_type") == "code":
                src = c.get("source", [])
                if isinstance(src, list):
                    lines.extend(src)
                elif isinstance(src, str):
                    lines.extend(src.splitlines())
        return [l.rstrip("\n") for l in lines]
    except Exception:
        return []


def compute_expected_relative_datasets_path(from_dir: Path, repo_root: Path) -> str:
    datasets_dir = repo_root / "datasets"
    rel = os.path.relpath(datasets_dir, start=from_dir)
    # Normalize to POSIX-style for consistency inside notebooks
    return Path(rel).as_posix()


def scan_lines_for_issues(file_path: Path, lines: List[str], repo_root: Path) -> List[Issue]:
    issues: List[Issue] = []

    # Patterns
    flat_file_pat = re.compile(
        r"datasets[\\/][^\\/\s]+\.(?:data|csv|txt|json|pkl|joblib)\b",
        re.IGNORECASE,
    )
    absolute_pat = re.compile(r"(^/|^[A-Za-z]:\\)")

    # Module 2/3 notebook-specific DATA_DIR rule only for notebook files
    is_notebook = file_path.suffix.lower() == ".ipynb"
    requires_data_dir_rule = False
    if is_notebook:
        try:
            # Determine if it's under module-2 or module-3 foundation
            parts = [p.lower() for p in file_path.parts]
            if "modules" in parts and "foundation" in parts and ("module-2" in parts or "module-3" in parts):
                requires_data_dir_rule = True
        except Exception:
            requires_data_dir_rule = False

    # Check per-line patterns
    for idx, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue

        # Flat file directly under datasets
        m_flat = flat_file_pat.search(text)
        if m_flat:
            # Determine severity contextually
            severity = "error"
            # Markdown files are often educational examples
            if file_path.suffix.lower() == ".md":
                severity = "warning"
            # Tests and documentation reproducibility modules may intentionally include anti-patterns
            fp = str(file_path.as_posix())
            if "test_" in file_path.name or "/tests/" in fp or "modules/project-dev/module-10/" in fp:
                severity = "warning"
            # Regex definition lines (indicative of validators) should not be treated as errors
            if re.search(r"r\"datasets\\/", text) or ".*\\.\\w+" in text:
                severity = "warning"

            issues.append(
                Issue(
                    file=str(file_path.relative_to(repo_root)),
                    line=idx,
                    kind=severity,
                    rule="datasets/flat-file",
                    message=(
                        "Dataset file referenced directly under 'datasets/'. "
                        "Organize into subfolders like 'datasets/<name>/<file>'."
                    ),
                    snippet=text[:240],
                )
            )

        # Absolute path usage
        if absolute_pat.search(text) and "datasets" in text:
            severity = "error"
            if file_path.suffix.lower() == ".md":
                severity = "warning"
            fp = str(file_path.as_posix())
            if "test_" in file_path.name or "/tests/" in fp or "modules/project-dev/module-10/" in fp:
                severity = "warning"
            issues.append(
                Issue(
                    file=str(file_path.relative_to(repo_root)),
                    line=idx,
                    kind=severity,
                    rule="datasets/absolute-path",
                    message=(
                        "Absolute path detected. Use relative paths from module "
                        "location and Path(...).resolve() patterns."
                    ),
                    snippet=text[:240],
                )
            )

    # Notebook-specific rule validation
    if requires_data_dir_rule:
        # Search for DATA_DIR assignment and .resolve()
        data_dir_lines = [(i, l) for i, l in enumerate(lines, start=1) if "DATA_DIR" in l and "Path(" in l]
        if not data_dir_lines:
            issues.append(
                Issue(
                    file=str(file_path.relative_to(repo_root)),
                    line=1,
                    kind="warning",
                    rule="notebook/data-dir-missing",
                    message=("Expected DATA_DIR = Path('../../../datasets').resolve() " "in Module 2/3 notebooks."),
                    snippet="",
                )
            )
        else:
            # Validate expected relative path component
            nb_dir = file_path.parent
            expected_rel = compute_expected_relative_datasets_path(nb_dir, repo_root)
            ok = False
            for ln, l in data_dir_lines:
                if ".resolve()" not in l:
                    issues.append(
                        Issue(
                            file=str(file_path.relative_to(repo_root)),
                            line=ln,
                            kind="warning",
                            rule="notebook/data-dir-no-resolve",
                            message=("DATA_DIR should use Path(...).resolve() to form an absolute path."),
                            snippet=l.strip()[:240],
                        )
                    )
                # Check that expected relative chunk is present
                if expected_rel in l.replace("\\", "/"):
                    ok = True
            if not ok:
                issues.append(
                    Issue(
                        file=str(file_path.relative_to(repo_root)),
                        line=data_dir_lines[0][0],
                        kind="warning",
                        rule="notebook/data-dir-relative-incorrect",
                        message=("DATA_DIR relative path does not match expected: " f"'{expected_rel}'."),
                        snippet=data_dir_lines[0][1].strip()[:240],
                    )
                )

    return issues


def scan_repository(repo_root: Path, include_dirs: Sequence[str], exts: Sequence[str]) -> Report:
    issues: List[Issue] = []
    scanned = 0

    for path in iter_files(repo_root, include_dirs, exts):
        scanned += 1
        if path.suffix.lower() == ".ipynb":
            lines = extract_notebook_sources(path)
        else:
            lines = read_text_lines(path)
        issues.extend(scan_lines_for_issues(path, lines, repo_root))

    return Report(root=str(repo_root), scanned_files=scanned, issues=issues)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify dataset path usage across notebooks, scripts, and docs. "
            "Checks flat files under datasets/, absolute paths, and Module 2/3 "
            "DATA_DIR conventions."
        )
    )

    parser.add_argument(
        "--paths",
        nargs="*",
        default=list(DEFAULT_INCLUDE_DIRS),
        help="Directories to include in the scan (default: modules docs datasets)",
    )
    parser.add_argument(
        "--exts",
        nargs="*",
        default=list(DEFAULT_FILE_EXTS),
        help="File extensions to include (default: .ipynb .py .md)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--fail-on",
        choices=["warning", "error", "never"],
        default="error",
        help=(
            "Fail on: 'error' (default) only errors trigger non-zero exit, "
            "'warning' (errors or warnings), 'never' always exit 0."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent

    include_dirs = [d for d in args.paths if d]
    exts = [e if e.startswith(".") else f".{e}" for e in args.exts]

    report = scan_repository(repo_root, include_dirs, exts)

    if args.format == "json":
        print(report.to_json())
    else:
        # Text summary
        print(f"Repo: {report.root}")
        print(f"Scanned files: {report.scanned_files}")
        print(f"Issues: {len(report.issues)}\n")
        for i in report.issues:
            print(f"[{i.kind.upper()}] {i.rule} - {i.file}:{i.line}\n  {i.message}\n  > {i.snippet}\n")

    # Determine exit code
    if args.fail_on == "never":
        return 0

    has_error = any(i.kind == "error" for i in report.issues)
    has_warning = any(i.kind == "warning" for i in report.issues)

    if args.fail_on == "error":
        return 2 if has_error else 0
    if args.fail_on == "warning":
        return 2 if (has_error or has_warning) else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
