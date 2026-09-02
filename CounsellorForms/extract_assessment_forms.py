"""CLI entry point for the assessment-form extraction pipeline.

Usage:
    python extract_assessment_forms.py [--forms DIR] [--out DIR] [--keep-names]

Defaults:
    --forms  auto-located beside the repo (e.g. ../CounsellorForms/Forms)
    --out    output/         (relative to this script)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eml_extractor.pipeline import run_pipeline


def _default_forms_root(base: Path) -> Path:
    """Locate the counsellor-form data directory.

    The raw ``.eml`` files contain PII and are intentionally kept OUT of the
    git repository, so the code and the data live in different places.
    Candidates are checked in order:

      1. ``<this script>/Forms``                  - data placed inside the repo
      2. ``<repo root>/../CounsellorForms/Forms`` - data beside the repo

    Returns the first candidate that exists, otherwise the first candidate so
    that ``main()`` can emit a clear "not found" message.
    """
    candidates = [base / "Forms", base.parent.parent / "CounsellorForms" / "Forms"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract & clean Jotform assessment data from .eml files."
    )
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--forms",
        type=Path,
        default=_default_forms_root(base),
        help="Root directory containing .eml files "
        "(default: auto-located beside the repo, e.g. ../CounsellorForms/Forms)",
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
        print(
            "Point --forms at your .eml data, or place the Jotform .eml files under "
            "CounsellorForms/Forms/.",
            file=sys.stderr,
        )
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
