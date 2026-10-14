"""Render Supplementary Table 3 from its editable definitions."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("glossary", root / "code/glossary.py")
    glossary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glossary)
    terms = glossary.load_glossary(root / "data/glossary.json")
    output = glossary.render_glossary(terms, args.output_dir / "Supplementary_Table_3_Glossary.pdf")
    print(output)


if __name__ == "__main__":
    main()
