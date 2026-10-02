from __future__ import annotations

import argparse
import json
from pathlib import Path

import amr
import penman

from src.evaluation.oldfile import (
    find_duplicate_variables,
    find_none_instances,
)


def is_smatch_parseable(amr_string: str) -> bool:
    """
    Check whether the AMR can be parsed by the parser
    used by the smatch package.
    """

    try:
        parsed = amr.AMR.parse_AMR_line(
            " ".join(amr_string.split())
        )
        return parsed is not None

    except Exception:
        return False


def check_gold_amr(amr_string: str) -> list[str]:
    """
    Check one gold AMR for structural problems.

    Returns an empty list if no problems are found.
    """

    errors = []

    # 1. Check duplicate variable definitions.
    duplicate_variables = find_duplicate_variables(
        amr_string
    )

    if duplicate_variables:
        errors.append(
            f"duplicate_variables: "
            f"{duplicate_variables}"
        )

    # 2. Check whether PENMAN can parse it.
    try:
        graph = penman.decode(amr_string)

    except Exception as exc:
        errors.append(
            f"penman_parse_error: {exc}"
        )

        # No point checking the graph itself
        # if PENMAN could not create one.
        graph = None

    # 3. Check for nodes without concepts.
    if graph is not None:

        none_instances = find_none_instances(
            graph
        )

        if none_instances:
            errors.append(
                f"none_instances: "
                f"{none_instances}"
            )

    # 4. Check whether Smatch can parse it.
    if not is_smatch_parseable(amr_string):
        errors.append(
            "smatch_parse_error"
        )

    return errors


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Check all gold AMRs for "
            "structural problems."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing gold_amr fields.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON file for saving "
            "problematic samples."
        ),
    )

    args = parser.parse_args()

    # -------------------------
    # Load dataset
    # -------------------------

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError(
            "Input JSON must contain a list "
            "of samples."
        )

    # -------------------------
    # Check gold AMRs
    # -------------------------

    problematic = []

    for sample in samples:

        interpretation_id = sample.get(
            "interpretation_id",
            "unknown",
        )

        gold_amr = sample.get("formalism")

        # Missing gold AMR
        if not gold_amr:

            problematic.append(
                {
                    "interpretation_id":
                        interpretation_id,
                    "errors": [
                        "missing_gold"
                    ],
                    "gold_amr": gold_amr,
                }
            )

            continue

        # Check the AMR
        errors = check_gold_amr(
            gold_amr
        )

        if errors:

            problematic.append(
                {
                    "interpretation_id":
                        interpretation_id,
                    "errors": errors,
                    "gold_amr": gold_amr,
                }
            )

    # -------------------------
    # Print summary
    # -------------------------

    total = len(samples)
    bad = len(problematic)
    good = total - bad

    print()
    print("Gold AMR Audit")
    print("=" * 50)
    print(f"Total:       {total}")
    print(f"Valid:       {good}")
    print(f"Problematic: {bad}")

    if total > 0:
        print(
            f"Valid rate:  "
            f"{good / total * 100:.2f}%"
        )

    # -------------------------
    # Print problematic AMRs
    # -------------------------

    for item in problematic:

        print()
        print("=" * 80)

        print(
            "ID:",
            item["interpretation_id"],
        )

        for error in item["errors"]:
            print(
                "ERROR:",
                error,
            )

        print("GOLD:")
        print(
            item["gold_amr"]
        )

    # -------------------------
    # Optionally save problems
    # -------------------------

    if args.output is not None:

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                problematic,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print(
            f"Problematic samples saved to: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()