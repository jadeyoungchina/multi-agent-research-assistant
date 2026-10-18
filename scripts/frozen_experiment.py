"""Create/verify the freeze, execute immutable repeats, or judge saved results."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.frozen import FREEZE_PATH, create_freeze, run_repeat, validate_freeze
from app.evaluation.test_judging import judge_repeats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("create", "verify", "generate", "judge", "analyze"))
    parser.add_argument("--freeze", type=Path, default=FREEZE_PATH)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/evaluation/real-v1-test")
    parser.add_argument("--repeat", choices=("all", "0", "1", "2"), default="all")
    parser.add_argument("--execute", action="store_true", help="make paid requests; otherwise validate only")
    parser.add_argument("--resolve-interrupted-as-failed", action="store_true",
                        help="retain an uncertain interrupted generation as failed; never retry it")
    args = parser.parse_args(argv)
    try:
        if args.operation == "create":
            record = create_freeze(args.freeze)
            print(f"Created {record['freeze_id']}; commit this record before paid execution.")
            return 0
        record = validate_freeze(args.freeze)
        repeats = list(range(3)) if args.repeat == "all" else [int(args.repeat)]
        if args.operation == "verify" or (args.operation in {"generate", "judge"} and not args.execute):
            print(f"Validated freeze {record['freeze_id']}; 480 planned units; no paid requests.")
            return 0
        if args.operation == "generate":
            for repeat in repeats:
                run_repeat(args.freeze, args.run_dir, repeat, resolve_interrupted=args.resolve_interrupted_as_failed)
        elif args.operation == "judge":
            entries = judge_repeats(args.freeze, args.run_dir, repeats)
            return 3 if any(e["status"] == "judge_error" for e in entries) else 0
        elif args.operation == "analyze":
            from app.evaluation.test_analysis import analyze
            result = analyze(args.freeze, args.run_dir)
            print(f"Analyzed {result['executed_units']}/480 units; primary complete pairs: {result['primary']['complete_pairs']}.")
        return 0
    except Exception as error:
        # Do not expose raw provider errors, credentials or account diagnostics.
        print(f"Frozen experiment stopped ({type(error).__name__}); no override of frozen inputs is allowed.", file=sys.stderr)
        if type(error) is ValueError and str(error) == "uncertain paid unit; use explicit interrupted-failure resolution":
            print("An interrupted unit is pending. Retain it as failed with --resolve-interrupted-as-failed; it will not be regenerated.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
