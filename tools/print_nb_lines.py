import json
from pathlib import Path
import sys


def main(nb_path: str, contains: str = "DATA_DIR") -> None:
    p = Path(nb_path)
    data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    cells = data.get("cells", [])
    matches = []
    for c in cells:
        if c.get("cell_type") == "code":
            src = c.get("source", [])
            if isinstance(src, str):
                src = src.splitlines()
            for line in src:
                if contains in line:
                    matches.append(line.rstrip("\n"))
    print("Found", len(matches), f"lines containing '{contains}':")
    for m in matches:
        print("-", m)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/print_nb_lines.py <notebook.ipynb> [substring]")
        sys.exit(1)
    nb = sys.argv[1]
    substr = sys.argv[2] if len(sys.argv) > 2 else "DATA_DIR"
    main(nb, substr)
