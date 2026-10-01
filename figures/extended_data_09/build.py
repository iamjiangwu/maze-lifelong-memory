"""Build Extended Data Figure 9 from included derived Source Data."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([
        sys.executable, "-B", str(repository / "reproduce.py"),
        "--figure", "ed9", *sys.argv[1:],
    ]))
