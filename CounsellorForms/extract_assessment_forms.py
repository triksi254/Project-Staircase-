"""CLI entry point for the assessment-form extraction pipeline.

Usage:
    python extract_assessment_forms.py [--forms DIR] [--out DIR] [--keep-names]

Defaults:
    --forms  Forms/          (relative to this script)
    --out    output/         (relative to this script)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eml_extractor.pipeline import run_pipeline


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract & clean Jotform assessment data from .eml files."
    )
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--forms",
        type=Path,
        default=base / "Forms",
        help="Root directory containing .eml files (default: Forms/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=base / "output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--keep-names",
        action="store_true",
        help="Keep counsellor names (NOT recommended; PII).",
    )
    args = parser.parse_args(argv)

    if not args.forms.exists():
        print(f"Forms directory not found: {args.forms}", file=sys.stderr)
        return 2

    summary = run_pipeline(
        forms_root=args.forms,
        out_dir=args.out,
        replace_names=not args.keep_names,
    )
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
