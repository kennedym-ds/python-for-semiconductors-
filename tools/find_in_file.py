from pathlib import Path
import sys


def main(path: str, needle: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    print(f"File size: {len(text)} bytes")
    count = text.count(needle)
    print(f"Occurrences of '{needle}': {count}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/find_in_file.py <path> <substring>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
