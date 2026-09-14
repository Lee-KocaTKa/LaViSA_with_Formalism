from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# Utilities
# ============================================================

def get_category(
    sample_id: str,
) -> str:
    """
    vp-100_1 -> vp
    coord-23_2 -> coord
    """
    return sample_id.split("-", 1)[0]


def get_ps_group_id(
    sample_id: str,
) -> str:
    """
    vp-100_1 -> vp-100
    vp-100_2 -> vp-100

    The final _1 / _2 / _3 identifies the interpretation.
    PS requires all interpretations belonging to the
    same original sample to be correct.
    """

    if "_" not in sample_id:
        return sample_id

    return sample_id.rsplit("_", 1)[0]


def safe_div(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def normalize_clean_choice(
    model_choice: object,
) -> str | None:
    """
    Automatically accept ONLY a clean answer:

        "1"
        "2"
        "3"

    Anything containing explanation or additional text
    must be manually reviewed.
    """

    if model_choice is None:
        return None

    choice = str(model_choice).strip()

    if choice in {"1", "2", "3"}:
        return choice

    return None


# ============================================================
# Manual review
# ============================================================

def manually_review_choice(
    sample: dict,
    current_index: int,
    total_manual: int,
) -> tuple[str | None, bool]:
    """
    Ask the user to manually interpret a verbose model response.

    Returns
    -------
    resolved_choice:
        "1", "2", "3", or None

    forced_wrong:
        True if user explicitly marks the response as wrong.
    """

    print()
    print("=" * 100)
    print(
        f"MANUAL REVIEW "
        f"[{current_index}/{total_manual}]"
    )
    print("=" * 100)

    print(
        f"Sample ID:       "
        f"{sample.get('sample_id')}"
    )

    print(
        f"Correct answer:  "
        f"{sample.get('correct_answer')}"
    )

    print()
    print("AMR:")
    print("-" * 100)

    print(
        sample.get(
            "amr",
            ""
        )
    )

    print()
    print("Image paths:")
    print("-" * 100)

    image_paths = sample.get(
        "image_path",
        []
    )

    if isinstance(
        image_paths,
        list,
    ):
        for i, path in enumerate(
            image_paths,
            start=1,
        ):
            print(
                f"{i}: {path}"
            )
    else:
        print(
            image_paths
        )

    print()
    print("MODEL OUTPUT:")
    print("-" * 100)

    print(
        sample.get(
            "model_choice",
            ""
        )
    )

    print()
    print("-" * 100)

    while True:

        judgment = input(
            "Interpret model output as "
            "[1/2/3], mark wrong [x], "
            "or quit [q]: "
        ).strip().lower()

        if judgment in {
            "1",
            "2",
            "3",
        }:
            return judgment, False

        if judgment == "x":
            return None, True

        if judgment == "q":
            raise KeyboardInterrupt(
                "Manual evaluation stopped by user."
            )

        print(
            "Please enter 1, 2, 3, x, or q."
        )


# ============================================================
# First pass: identify manual-review samples
# ============================================================

def find_manual_review_indices(
    samples: list[dict],
) -> list[int]:

    indices = []

    for i, sample in enumerate(
        samples
    ):

        # API/generation failure:
        # no need for manual interpretation.
        if sample.get(
            "error"
        ) is not None:
            continue

        model_choice = sample.get(
            "model_choice"
        )

        clean_choice = (
            normalize_clean_choice(
                model_choice
            )
        )

        if clean_choice is None:
            indices.append(
                i
            )

    return indices


# ============================================================
# Evaluate samples
# ============================================================

def evaluate_samples(
    samples: list[dict],
) -> list[dict]:

    evaluated = [
        dict(sample)
        for sample in samples
    ]

    manual_indices = (
        find_manual_review_indices(
            evaluated
        )
    )

    total_manual = len(
        manual_indices
    )

    print(
        f"Total samples: "
        f"{len(evaluated)}"
    )

    print(
        f"Samples requiring manual review: "
        f"{total_manual}"
    )

    manual_counter = 0

    for i, sample in enumerate(
        evaluated
    ):

        sample_id = sample.get(
            "sample_id",
            "unknown",
        )

        category = get_category(
            sample_id
        )

        ps_group_id = get_ps_group_id(
            sample_id
        )

        sample[
            "category"
        ] = category

        sample[
            "ps_group_id"
        ] = ps_group_id

        correct_answer = str(
            sample.get(
                "correct_answer",
                ""
            )
        ).strip()

        error = sample.get(
            "error"
        )

        # ----------------------------------------------------
        # Generation/API error
        # ----------------------------------------------------

        if error is not None:

            sample[
                "resolved_model_choice"
            ] = None

            sample[
                "choice_source"
            ] = "generation_error"

            sample[
                "is_correct"
            ] = False

            continue

        # ----------------------------------------------------
        # Clean 1 / 2 / 3
        # ----------------------------------------------------

        clean_choice = (
            normalize_clean_choice(
                sample.get(
                    "model_choice"
                )
            )
        )

        if clean_choice is not None:

            sample[
                "resolved_model_choice"
            ] = clean_choice

            sample[
                "choice_source"
            ] = "automatic"

            sample[
                "is_correct"
            ] = (
                clean_choice
                == correct_answer
            )

            continue

        # ----------------------------------------------------
        # Manual interpretation
        # ----------------------------------------------------

        manual_counter += 1

        resolved_choice, forced_wrong = (
            manually_review_choice(
                sample,
                current_index=manual_counter,
                total_manual=total_manual,
            )
        )

        sample[
            "resolved_model_choice"
        ] = resolved_choice

        if forced_wrong:

            sample[
                "choice_source"
            ] = "manual_wrong"

            sample[
                "is_correct"
            ] = False

        else:

            sample[
                "choice_source"
            ] = "manual"

            sample[
                "is_correct"
            ] = (
                resolved_choice
                == correct_answer
            )

    return evaluated


# ============================================================
# PT
# ============================================================

def compute_pt(
    samples: list[dict],
) -> dict:

    total = len(
        samples
    )

    correct = sum(
        1
        for sample in samples
        if sample.get(
            "is_correct",
            False,
        )
    )

    automatic = sum(
        1
        for sample in samples
        if sample.get(
            "choice_source"
        ) == "automatic"
    )

    manual = sum(
        1
        for sample in samples
        if sample.get(
            "choice_source"
        ) == "manual"
    )

    manual_wrong = sum(
        1
        for sample in samples
        if sample.get(
            "choice_source"
        ) == "manual_wrong"
    )

    generation_errors = sum(
        1
        for sample in samples
        if sample.get(
            "choice_source"
        ) == "generation_error"
    )

    return {
        "total_trials": total,

        "correct_trials": correct,

        "pt_accuracy": safe_div(
            correct,
            total,
        ),

        "automatic_choices": automatic,

        "manual_choices": manual,

        "manual_wrong": manual_wrong,

        "generation_errors":
            generation_errors,
    }


# ============================================================
# PS
# ============================================================

def compute_ps(
    samples: list[dict],
) -> dict:
    """
    PS is calculated over the original sample.

    Example:

        vp-100_1 -> correct
        vp-100_2 -> correct

        => vp-100 is PS-correct.

    If either fails:

        vp-100_1 -> correct
        vp-100_2 -> wrong

        => vp-100 is PS-wrong.
    """

    groups = defaultdict(
        list
    )

    for sample in samples:

        group_id = sample[
            "ps_group_id"
        ]

        groups[
            group_id
        ].append(
            sample
        )

    total_groups = len(
        groups
    )

    correct_groups = 0

    group_details = []

    for group_id, group_samples in sorted(
        groups.items()
    ):

        all_correct = all(
            sample.get(
                "is_correct",
                False,
            )
            for sample in group_samples
        )

        if all_correct:
            correct_groups += 1

        category = (
            group_samples[0][
                "category"
            ]
        )

        group_details.append(
            {
                "group_id":
                    group_id,

                "category":
                    category,

                "num_trials":
                    len(
                        group_samples
                    ),

                "all_correct":
                    all_correct,

                "trials": [
                    {
                        "sample_id":
                            sample.get(
                                "sample_id"
                            ),

                        "correct_answer":
                            sample.get(
                                "correct_answer"
                            ),

                        "resolved_model_choice":
                            sample.get(
                                "resolved_model_choice"
                            ),

                        "choice_source":
                            sample.get(
                                "choice_source"
                            ),

                        "is_correct":
                            sample.get(
                                "is_correct"
                            ),
                    }
                    for sample
                    in group_samples
                ],
            }
        )

    return {
        "total_groups":
            total_groups,

        "correct_groups":
            correct_groups,

        "ps_accuracy":
            safe_div(
                correct_groups,
                total_groups,
            ),

        "groups":
            group_details,
    }


# ============================================================
# Category metrics
# ============================================================

def compute_by_category(
    samples: list[dict],
) -> dict:

    categories = defaultdict(
        list
    )

    for sample in samples:

        categories[
            sample[
                "category"
            ]
        ].append(
            sample
        )

    results = {}

    for category in sorted(
        categories
    ):

        category_samples = (
            categories[
                category
            ]
        )

        pt = compute_pt(
            category_samples
        )

        ps = compute_ps(
            category_samples
        )

        results[
            category
        ] = {
            "pt": pt,

            "ps": {
                "total_groups":
                    ps[
                        "total_groups"
                    ],

                "correct_groups":
                    ps[
                        "correct_groups"
                    ],

                "ps_accuracy":
                    ps[
                        "ps_accuracy"
                    ],
            },
        }

    return results


# ============================================================
# Error analysis
# ============================================================

def compute_choice_confusion(
    samples: list[dict],
) -> dict:
    """
    Count how often each gold option is predicted as each option.

    Useful when categories have 2 or 3 candidate images.
    """

    matrix = defaultdict(
        lambda: defaultdict(
            int
        )
    )

    forced_wrong = 0
    generation_error = 0

    for sample in samples:

        gold = str(
            sample.get(
                "correct_answer",
                ""
            )
        ).strip()

        pred = sample.get(
            "resolved_model_choice"
        )

        source = sample.get(
            "choice_source"
        )

        if source == "generation_error":
            generation_error += 1
            continue

        if pred is None:
            forced_wrong += 1
            continue

        matrix[
            gold
        ][
            pred
        ] += 1

    return {
        "matrix": {
            gold: dict(
                predictions
            )
            for gold, predictions
            in matrix.items()
        },

        "manual_wrong":
            forced_wrong,

        "generation_error":
            generation_error,
    }


# ============================================================
# Printing
# ============================================================

def print_summary(
    overall_pt: dict,
    overall_ps: dict,
    by_category: dict,
) -> None:

    print()
    print(
        "=" * 85
    )

    print(
        "OVERALL"
    )

    print(
        "=" * 85
    )

    print(
        f"PT Accuracy:   "
        f"{overall_pt['pt_accuracy'] * 100:.2f}% "
        f"({overall_pt['correct_trials']}/"
        f"{overall_pt['total_trials']})"
    )

    print(
        f"PS Accuracy:   "
        f"{overall_ps['ps_accuracy'] * 100:.2f}% "
        f"({overall_ps['correct_groups']}/"
        f"{overall_ps['total_groups']})"
    )

    print()

    print(
        f"Automatic:     "
        f"{overall_pt['automatic_choices']}"
    )

    print(
        f"Manual judged: "
        f"{overall_pt['manual_choices']}"
    )

    print(
        f"Manual wrong:  "
        f"{overall_pt['manual_wrong']}"
    )

    print(
        f"Errors:        "
        f"{overall_pt['generation_errors']}"
    )

    print()
    print(
        "=" * 85
    )

    print(
        "BY CATEGORY"
    )

    print(
        "=" * 85
    )

    header = (
        f"{'Category':<12}"
        f"{'PT N':>8}"
        f"{'PT Acc':>14}"
        f"{'PS N':>8}"
        f"{'PS Acc':>14}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for category, data in (
        by_category.items()
    ):

        pt = data[
            "pt"
        ]

        ps = data[
            "ps"
        ]

        print(
            f"{category:<12}"

            f"{pt['total_trials']:>8}"

            f"{pt['pt_accuracy'] * 100:>13.2f}%"

            f"{ps['total_groups']:>8}"

            f"{ps['ps_accuracy'] * 100:>13.2f}%"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AMR-to-image matching "
            "with PT and PS accuracy."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:

        samples = json.load(
            f
        )

    if not isinstance(
        samples,
        list,
    ):
        raise ValueError(
            "Input JSON must contain a list."
        )

    # --------------------------------------------------------
    # Evaluation + manual review
    # --------------------------------------------------------

    try:

        evaluated_samples = (
            evaluate_samples(
                samples
            )
        )

    except KeyboardInterrupt:

        print()
        print(
            "Evaluation interrupted."
        )

        print(
            "No final output was written."
        )

        return

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_pt = compute_pt(
        evaluated_samples
    )

    overall_ps_full = compute_ps(
        evaluated_samples
    )

    overall_ps = {
        "total_groups":
            overall_ps_full[
                "total_groups"
            ],

        "correct_groups":
            overall_ps_full[
                "correct_groups"
            ],

        "ps_accuracy":
            overall_ps_full[
                "ps_accuracy"
            ],
    }

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    by_category = (
        compute_by_category(
            evaluated_samples
        )
    )

    # --------------------------------------------------------
    # Choice confusion
    # --------------------------------------------------------

    confusion = (
        compute_choice_confusion(
            evaluated_samples
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output = {
        "summary": {
            "overall": {
                "pt":
                    overall_pt,

                "ps":
                    overall_ps,
            },

            "by_category":
                by_category,

            "choice_confusion":
                confusion,
        },

        "ps_groups":
            overall_ps_full[
                "groups"
            ],

        "samples":
            evaluated_samples,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(
        overall_pt,
        overall_ps,
        by_category,
    )

    print()
    print(
        f"Saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()