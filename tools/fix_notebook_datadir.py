import json
import os
from pathlib import Path
import sys
from typing import List


def compute_expected_rel(nb_path: Path, datasets_dirname: str = "datasets") -> str:
    nb_dir = nb_path.parent
    cur = nb_dir
    while cur != cur.parent:
        if (cur / datasets_dirname).exists():
            rel = os.path.relpath(cur / datasets_dirname, start=nb_dir)
            return Path(rel).as_posix()
        cur = cur.parent
    return "../../../datasets"


def ensure_datadir_cell(nb_path: Path) -> bool:
    data = json.loads(nb_path.read_text(encoding="utf-8", errors="ignore"))
    cells: List[dict] = data.get("cells", [])

    expected_rel = compute_expected_rel(nb_path)
    desired_lines = [
        "from pathlib import Path\n",
        f"DATA_DIR = Path('{expected_rel}').resolve()\n",
    ]

    # Try update existing
    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if isinstance(src, str):
            src = src.splitlines(keepends=True)
        joined = "".join(src)
        if "DATA_DIR" in joined and "Path(" in joined:
            new_src = []
            inserted_import = any("from pathlib import Path" in s for s in src)
            for line in src:
                if "DATA_DIR" in line and "Path(" in line:
                    new_src.append(desired_lines[1])
                else:
                    new_src.append(line)
            if not inserted_import:
                new_src.insert(0, desired_lines[0])
            c["source"] = new_src
            nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
            return True

    # Insert new cell after first code cell or at index 1
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": desired_lines,
    }
    insert_idx = 1
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code":
            insert_idx = i + 1
            break
    cells.insert(insert_idx, new_cell)
    data["cells"] = cells
    nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/fix_notebook_datadir.py <notebook.ipynb> [more.ipynb ...]")
        return 1
    ok = True
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"Skip missing: {p}")
            ok = False
            continue
        ensure_datadir_cell(p)
        print(f"Updated: {p}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
