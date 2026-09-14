from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# Utilities
# ============================================================

def normalize_yes_no(
    answer: str | None,
) -> bool | None:
    """
    Convert model response into boolean.

    Yes -> True
    No  -> False

    Invalid/unrecognized output -> None
    """

    if answer is None:
        return None

    normalized = answer.strip().lower()

    if normalized == "yes":
        return True

    if normalized == "no":
        return False

    return None


def get_category_from_image_path(
    image_path: str,
) -> str:
    """
    Example:

    /.../images/vp/vp-100-a-i.png
        -> vp
    """

    path = Path(image_path)

    # parent directory:
    # .../images/vp/
    return path.parent.name


def safe_div(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# Per-trial evaluation
# ============================================================

def evaluate_trials(
    samples: list[dict],
) -> list[dict]:

    evaluated = []

    for sample in samples:

        item = dict(sample)

        image_path = item.get(
            "image_path",
            ""
        )

        category = get_category_from_image_path(
            image_path
        )

        item["category"] = category

        gold = item.get(
            "correct_or_incorrect"
        )

        answer = item.get(
            "yes_or_no"
        )

        existing_error = item.get(
            "error"
        )

        prediction = normalize_yes_no(
            answer
        )

        item["prediction_bool"] = prediction

        # ----------------------------------------------------
        # Generation/API error
        # ----------------------------------------------------

        if existing_error is not None:

            item["is_correct"] = False
            item["evaluation_status"] = "generation_error"

            evaluated.append(
                item
            )

            continue

        # ----------------------------------------------------
        # Invalid Yes/No
        # ----------------------------------------------------

        if prediction is None:

            item["is_correct"] = False
            item["evaluation_status"] = "invalid_answer"

            evaluated.append(
                item
            )

            continue

        # ----------------------------------------------------
        # Normal evaluation
        # ----------------------------------------------------

        is_correct = (
            prediction == gold
        )

        item["is_correct"] = is_correct

        if gold is True and prediction is True:
            status = "true_positive"

        elif gold is False and prediction is False:
            status = "true_negative"

        elif gold is False and prediction is True:
            status = "false_positive"

        elif gold is True and prediction is False:
            status = "false_negative"

        else:
            status = "unknown"

        item["evaluation_status"] = status

        evaluated.append(
            item
        )

    return evaluated


# ============================================================
# PT statistics
# ============================================================

def compute_pt_statistics(
    samples: list[dict],
) -> dict:

    total = len(samples)

    correct = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    invalid = 0
    generation_errors = 0

    positive_gold = 0
    negative_gold = 0

    for sample in samples:

        gold = sample.get(
            "correct_or_incorrect"
        )

        status = sample.get(
            "evaluation_status"
        )

        if gold is True:
            positive_gold += 1

        elif gold is False:
            negative_gold += 1

        if sample.get(
            "is_correct",
            False
        ):
            correct += 1

        if status == "true_positive":
            tp += 1

        elif status == "true_negative":
            tn += 1

        elif status == "false_positive":
            fp += 1

        elif status == "false_negative":
            fn += 1

        elif status == "invalid_answer":
            invalid += 1

        elif status == "generation_error":
            generation_errors += 1

    return {
        "total_trials": total,

        "correct_trials": correct,

        "pt_accuracy": safe_div(
            correct,
            total,
        ),

        "true_positive": tp,
        "true_negative": tn,

        "false_positive": fp,
        "false_negative": fn,

        "positive_gold": positive_gold,
        "negative_gold": negative_gold,

        # Among actually-negative examples,
        # how often did model incorrectly say Yes?
        "false_positive_rate": safe_div(
            fp,
            negative_gold,
        ),

        # Among actually-positive examples,
        # how often did model incorrectly say No?
        "false_negative_rate": safe_div(
            fn,
            positive_gold,
        ),

        "invalid_answers": invalid,
        "generation_errors": generation_errors,
    }


# ============================================================
# PS statistics
# ============================================================

def compute_ps_statistics(
    samples: list[dict],
) -> dict:
    """
    PS = Per-Sentence / Per-Structure accuracy.

    Here the grouping key is gold_amr.

    Every trial corresponding to the SAME gold AMR must
    be correct for that AMR to count as correct.

    Example:

        AMR A + image A -> correct
        AMR A + image B -> correct

    => PS correct

    If even one fails:

        AMR A + image A -> correct
        AMR A + image B -> incorrect

    => PS incorrect
    """

    grouped = defaultdict(
        list
    )

    for sample in samples:

        gold_amr = sample.get(
            "gold_amr"
        )

        grouped[
            gold_amr
        ].append(
            sample
        )

    total_groups = len(
        grouped
    )

    correct_groups = 0

    group_results = []

    for gold_amr, group in grouped.items():

        all_correct = all(
            sample.get(
                "is_correct",
                False
            )
            for sample in group
        )

        if all_correct:
            correct_groups += 1

        categories = {
            sample[
                "category"
            ]
            for sample in group
        }

        # Normally one category only.
        category = (
            next(iter(categories))
            if len(categories) == 1
            else "mixed"
        )

        group_results.append(
            {
                "gold_amr": gold_amr,
                "category": category,
                "num_trials": len(group),
                "all_correct": all_correct,

                "trial_results": [
                    {
                        "image_path":
                            sample.get(
                                "image_path"
                            ),

                        "gold":
                            sample.get(
                                "correct_or_incorrect"
                            ),

                        "prediction":
                            sample.get(
                                "prediction_bool"
                            ),

                        "is_correct":
                            sample.get(
                                "is_correct"
                            ),

                        "status":
                            sample.get(
                                "evaluation_status"
                            ),
                    }
                    for sample in group
                ],
            }
        )

    return {
        "total_groups": total_groups,

        "correct_groups": correct_groups,

        "ps_accuracy": safe_div(
            correct_groups,
            total_groups,
        ),

        "groups": group_results,
    }


# ============================================================
# Category aggregation
# ============================================================

def compute_by_category(
    samples: list[dict],
) -> dict:

    grouped = defaultdict(
        list
    )

    for sample in samples:

        category = sample[
            "category"
        ]

        grouped[
            category
        ].append(
            sample
        )

    results = {}

    for category in sorted(
        grouped
    ):

        category_samples = grouped[
            category
        ]

        pt = compute_pt_statistics(
            category_samples
        )

        ps = compute_ps_statistics(
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
# Printing
# ============================================================

def print_overall(
    pt: dict,
    ps: dict,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        "OVERALL AMR VERIFICATION"
    )

    print(
        "=" * 80
    )

    print(
        f"PT Accuracy:          "
        f"{pt['pt_accuracy'] * 100:.2f}% "
        f"({pt['correct_trials']}/"
        f"{pt['total_trials']})"
    )

    print(
        f"PS Accuracy:          "
        f"{ps['ps_accuracy'] * 100:.2f}% "
        f"({ps['correct_groups']}/"
        f"{ps['total_groups']})"
    )

    print()

    print(
        f"True Positive:        "
        f"{pt['true_positive']}"
    )

    print(
        f"True Negative:        "
        f"{pt['true_negative']}"
    )

    print(
        f"False Positive:       "
        f"{pt['false_positive']}"
    )

    print(
        f"False Negative:       "
        f"{pt['false_negative']}"
    )

    print()

    print(
        f"False Positive Rate:  "
        f"{pt['false_positive_rate'] * 100:.2f}%"
    )

    print(
        f"False Negative Rate:  "
        f"{pt['false_negative_rate'] * 100:.2f}%"
    )

    print()

    print(
        f"Invalid answers:      "
        f"{pt['invalid_answers']}"
    )

    print(
        f"Generation errors:    "
        f"{pt['generation_errors']}"
    )


def print_by_category(
    results: dict,
) -> None:

    print()
    print(
        "=" * 110
    )

    print(
        "BY CATEGORY"
    )

    print(
        "=" * 110
    )

    header = (
        f"{'Cat':<10}"
        f"{'PT':>12}"
        f"{'PS':>12}"
        f"{'TP':>8}"
        f"{'TN':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
        f"{'FPR':>12}"
        f"{'FNR':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for category, data in results.items():

        pt = data[
            "pt"
        ]

        ps = data[
            "ps"
        ]

        row = (
            f"{category:<10}"

            f"{pt['pt_accuracy'] * 100:>11.2f}%"

            f"{ps['ps_accuracy'] * 100:>11.2f}%"

            f"{pt['true_positive']:>8}"

            f"{pt['true_negative']:>8}"

            f"{pt['false_positive']:>8}"

            f"{pt['false_negative']:>8}"

            f"{pt['false_positive_rate'] * 100:>11.2f}%"

            f"{pt['false_negative_rate'] * 100:>11.2f}%"
        )

        print(
            row
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AMR-image verification "
            "using PT and PS accuracy."
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

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

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
    # Evaluate trials
    # --------------------------------------------------------

    evaluated_samples = evaluate_trials(
        samples
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_pt = (
        compute_pt_statistics(
            evaluated_samples
        )
    )

    overall_ps_full = (
        compute_ps_statistics(
            evaluated_samples
        )
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
    # By category
    # --------------------------------------------------------

    by_category = (
        compute_by_category(
            evaluated_samples
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "summary": {
            "overall": {
                "pt": overall_pt,
                "ps": overall_ps,
            },

            "by_category":
                by_category,
        },

        # Useful for inspecting exactly which AMRs
        # failed PS.
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

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_overall(
        overall_pt,
        overall_ps,
    )

    print_by_category(
        by_category
    )

    print()
    print(
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()