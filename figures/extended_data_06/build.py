"""Build Extended Data Figure 6 from included derived Source Data."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[2]
    raise SystemExit(subprocess.call([
        sys.executable, "-B", str(repository / "reproduce.py"),
        "--figure", "ed6", *sys.argv[1:],
    ]))
