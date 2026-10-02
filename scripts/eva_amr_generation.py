from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import amr
import penman
import smatch
from tqdm import tqdm

from src.evaluation.oldfile import (
    compute_srl_f1_from_graphs,
    encode_graph,
    find_duplicate_variables,
    find_none_instances,
    normalize_graph_wsd,
)

from src.paths import AUGMENTED_JSONS


# ============================================================
# Category utilities
# ============================================================

def get_category(sample: dict) -> str:
    """
    Example:
        vp-100-a -> vp
    """
    return sample["interpretation_id"].split("-", 1)[0]


def group_samples(
    samples: list[dict],
) -> dict[str, list[dict]]:

    groups = defaultdict(list)

    for sample in samples:
        category = get_category(sample)
        groups[category].append(sample)

    return groups


# ============================================================
# Smatch parsing
# ============================================================

def is_smatch_parseable(
    amr_string: str,
) -> bool:
    """
    Check whether the AMR can be parsed by the parser
    used by the Smatch implementation.
    """

    try:
        parsed = amr.AMR.parse_AMR_line(
            " ".join(amr_string.split())
        )
        return parsed is not None

    except Exception:
        return False


# ============================================================
# Smatch
# ============================================================

def compute_smatch(
    gold_amrs: list[str],
    pred_amrs: list[str],
) -> float | None:
    """
    Compute Smatch F1.

    Returns None if Smatch cannot parse or score the AMRs.

    Single sample:
        compute_smatch([gold], [pred])

    Corpus:
        compute_smatch(gold_amrs, pred_amrs)
    """

    if not gold_amrs:
        return None

    if len(gold_amrs) != len(pred_amrs):
        raise ValueError(
            "Gold and prediction counts do not match: "
            f"{len(gold_amrs)} vs {len(pred_amrs)}"
        )

    # Check every AMR before passing them to Smatch.
    for gold_amr, pred_amr in zip(
        gold_amrs,
        pred_amrs,
    ):
        if not is_smatch_parseable(gold_amr):
            return None

        if not is_smatch_parseable(pred_amr):
            return None

    gold_stream = io.StringIO(
        "\n\n".join(gold_amrs) + "\n\n"
    )

    pred_stream = io.StringIO(
        "\n\n".join(pred_amrs) + "\n\n"
    )

    try:
        scores = list(
            smatch.score_amr_pairs(
                gold_stream,
                pred_stream,
            )
        )

    except Exception:
        return None

    if not scores:
        return None

    # precision, recall, f1
    return scores[-1][2]


# ============================================================
# Per-sample evaluation
# ============================================================

def evaluate_sample(
    sample: dict,
) -> dict:

    sample = dict(sample)

    sample["valid"] = False
    sample["invalid_reason"] = None
    sample["metric_error"] = None

    sample["metrics"] = {
        "smatch": None,
        "smatch_no_wsd": None,
        "srl": None,
        "srl_no_wsd": None,
    }

    interpretation_id = sample.get(
        "interpretation_id",
        "unknown",
    )

    gold_amr = sample.get("gold_amr")
    pred_amr = sample.get("generated_amr")

    # --------------------------------------------------------
    # Gold existence
    # --------------------------------------------------------

    if not gold_amr:
        raise ValueError(
            f"Missing gold AMR: {interpretation_id}"
        )

    # --------------------------------------------------------
    # Prediction existence
    # --------------------------------------------------------

    if not pred_amr:
        sample["invalid_reason"] = (
            "missing_prediction"
        )
        return sample

    if sample.get("error") is not None:
        sample["invalid_reason"] = (
            "generation_error"
        )
        return sample

    # --------------------------------------------------------
    # Duplicate variable definitions
    # --------------------------------------------------------

    if find_duplicate_variables(pred_amr):
        sample["invalid_reason"] = (
            "duplicate_variables"
        )
        return sample

    # --------------------------------------------------------
    # Parse gold with PENMAN
    #
    # Gold failure = reference/dataset problem.
    # --------------------------------------------------------

    try:
        gold_graph = penman.decode(
            gold_amr
        )

    except Exception as exc:
        raise ValueError(
            "Gold PENMAN parse error: "
            f"{interpretation_id}"
        ) from exc

    # --------------------------------------------------------
    # Parse prediction with PENMAN
    #
    # Prediction failure = invalid model output.
    # --------------------------------------------------------

    try:
        pred_graph = penman.decode(
            pred_amr
        )

    except Exception as exc:
        sample["invalid_reason"] = (
            "penman_parse_error"
        )

        sample["metric_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return sample

    # --------------------------------------------------------
    # Missing instance concepts
    # --------------------------------------------------------

    if find_none_instances(pred_graph):
        sample["invalid_reason"] = (
            "none_instance"
        )
        return sample

    # --------------------------------------------------------
    # Gold Smatch parsing
    #
    # Again: gold failure = dataset problem.
    # --------------------------------------------------------

    if not is_smatch_parseable(gold_amr):
        raise ValueError(
            "Gold Smatch parse error: "
            f"{interpretation_id}"
        )

    # --------------------------------------------------------
    # Prediction Smatch parsing
    # --------------------------------------------------------

    if not is_smatch_parseable(pred_amr):
        sample["invalid_reason"] = (
            "smatch_parse_error"
        )
        return sample

    # ========================================================
    # Metric processing
    #
    # From this point onward, strange generated graphs can
    # still break normalization, encoding, or scoring.
    #
    # Such failures should invalidate this prediction rather
    # than terminate the entire evaluation.
    # ========================================================

    try:

        # ----------------------------------------------------
        # Standard Smatch
        # ----------------------------------------------------

        smatch_score = compute_smatch(
            [gold_amr],
            [pred_amr],
        )

        if smatch_score is None:
            sample["invalid_reason"] = (
                "smatch_score_error"
            )
            return sample

        # ----------------------------------------------------
        # Construct No-WSD graphs
        # ----------------------------------------------------

        gold_no_wsd = encode_graph(
            normalize_graph_wsd(
                gold_graph
            )
        )

        pred_no_wsd = encode_graph(
            normalize_graph_wsd(
                pred_graph
            )
        )

        # ----------------------------------------------------
        # No-WSD Smatch
        # ----------------------------------------------------

        smatch_no_wsd = compute_smatch(
            [gold_no_wsd],
            [pred_no_wsd],
        )

        if smatch_no_wsd is None:
            sample["invalid_reason"] = (
                "no_wsd_smatch_error"
            )
            return sample

        # ----------------------------------------------------
        # Coordination-normalized SRL F1
        # ----------------------------------------------------

        srl = compute_srl_f1_from_graphs(
            gold_graph,
            pred_graph,
        )

        # ----------------------------------------------------
        # Coordination-normalized SRL F1, No-WSD
        # ----------------------------------------------------

        srl_no_wsd = (
            compute_srl_f1_from_graphs(
                gold_graph,
                pred_graph,
                remove_wsd=True,
            )
        )

    except Exception as exc:

        sample["invalid_reason"] = (
            "metric_processing_error"
        )

        sample["metric_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        return sample

    # --------------------------------------------------------
    # Only save metrics after ALL metric processing succeeds.
    # --------------------------------------------------------

    sample["metrics"]["smatch"] = (
        smatch_score
    )

    sample["metrics"]["smatch_no_wsd"] = (
        smatch_no_wsd
    )

    sample["metrics"]["srl"] = (
        srl.f1
    )

    sample["metrics"]["srl_no_wsd"] = (
        srl_no_wsd.f1
    )

    sample["valid"] = True

    return sample


# ============================================================
# Aggregation
# ============================================================

def aggregate_group(
    samples: list[dict],
) -> dict:
    """
    Aggregate one collection of samples.

    Smatch:
        corpus-level Smatch over valid samples.

    SRL:
        macro-average of per-sample SRL F1.

    Invalid predictions:
        included in valid-rate denominator;
        excluded from semantic metrics.
    """

    total = len(samples)

    valid_samples = [
        sample
        for sample in samples
        if sample["valid"]
    ]

    valid_count = len(valid_samples)

    valid_rate = (
        valid_count / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # SRL macro-average
    # --------------------------------------------------------

    srl_scores = [
        sample["metrics"]["srl"]
        for sample in valid_samples
    ]

    srl_no_wsd_scores = [
        sample["metrics"]["srl_no_wsd"]
        for sample in valid_samples
    ]

    srl = (
        sum(srl_scores)
        / len(srl_scores)
        if srl_scores
        else 0.0
    )

    srl_no_wsd = (
        sum(srl_no_wsd_scores)
        / len(srl_no_wsd_scores)
        if srl_no_wsd_scores
        else 0.0
    )

    # --------------------------------------------------------
    # Corpus Smatch
    # --------------------------------------------------------

    gold_amrs = [
        sample["gold_amr"]
        for sample in valid_samples
    ]

    pred_amrs = [
        sample["generated_amr"]
        for sample in valid_samples
    ]

    smatch_score = compute_smatch(
        gold_amrs,
        pred_amrs,
    )

    if valid_samples and smatch_score is None:
        raise RuntimeError(
            "Corpus Smatch failed even though "
            "all individual samples were valid."
        )

    if smatch_score is None:
        smatch_score = 0.0

    # --------------------------------------------------------
    # Corpus No-WSD Smatch
    # --------------------------------------------------------

    gold_no_wsd = []
    pred_no_wsd = []

    for sample in valid_samples:

        gold_graph = penman.decode(
            sample["gold_amr"]
        )

        pred_graph = penman.decode(
            sample["generated_amr"]
        )

        gold_no_wsd.append(
            encode_graph(
                normalize_graph_wsd(
                    gold_graph
                )
            )
        )

        pred_no_wsd.append(
            encode_graph(
                normalize_graph_wsd(
                    pred_graph
                )
            )
        )

    smatch_no_wsd = compute_smatch(
        gold_no_wsd,
        pred_no_wsd,
    )

    if (
        valid_samples
        and smatch_no_wsd is None
    ):
        raise RuntimeError(
            "Corpus No-WSD Smatch failed even though "
            "all individual samples were valid."
        )

    if smatch_no_wsd is None:
        smatch_no_wsd = 0.0

    return {
        "n": total,
        "valid": valid_count,
        "valid_rate": valid_rate,
        "smatch": smatch_score,
        "smatch_no_wsd": smatch_no_wsd,
        "srl": srl,
        "srl_no_wsd": srl_no_wsd,
    }


def aggregate(
    samples: list[dict],
) -> dict:

    result = {
        "overall": aggregate_group(
            samples
        ),
        "by_category": {},
    }

    groups = group_samples(
        samples
    )

    for (
        category,
        category_samples,
    ) in groups.items():

        result[
            "by_category"
        ][category] = aggregate_group(
            category_samples
        )

    return result


# ============================================================
# Invalid reason statistics
# ============================================================

def count_invalid_reasons(
    samples: list[dict],
) -> dict[str, int]:

    counts = defaultdict(int)

    for sample in samples:

        if sample["valid"]:
            continue

        reason = sample[
            "invalid_reason"
        ]

        if reason is None:
            reason = "unknown"

        counts[reason] += 1

    return dict(
        sorted(counts.items())
    )


# ============================================================
# Printing
# ============================================================

def print_summary(
    summary: dict,
) -> None:

    print()

    print(
        f"{'Category':<10}"
        f"{'N':>8}"
        f"{'Valid':>10}"
        f"{'Smatch':>10}"
        f"{'No-WSD':>10}"
        f"{'SRL':>10}"
        f"{'SRL-NW':>10}"
    )

    print("-" * 68)

    rows = {
        "all": summary["overall"],
        **summary["by_category"],
    }

    for category, result in rows.items():

        print(
            f"{category:<10}"
            f"{result['n']:>8}"
            f"{result['valid_rate'] * 100:>9.2f}%"
            f"{result['smatch'] * 100:>9.2f}%"
            f"{result['smatch_no_wsd'] * 100:>9.2f}%"
            f"{result['srl'] * 100:>9.2f}%"
            f"{result['srl_no_wsd'] * 100:>9.2f}%"
        )


def print_invalid_reasons(
    invalid_reasons: dict[str, int],
) -> None:

    print()
    print("Invalid generated AMRs")
    print("-" * 40)

    if not invalid_reasons:
        print("None")
        return

    for reason, count in invalid_reasons.items():

        print(
            f"{reason:<30}"
            f"{count:>6}"
        )


# ============================================================
# Replace old gold references
# ============================================================

def replace_gold_references(
    samples: list[dict],
    revised_reference: list[dict],
) -> None:
    """
    Replace gold_amr in the generation output with the
    revised reference formalism.

    Every generated sample must have a revised reference.
    """

    reference_by_id = {
        ref["interpretation_id"]:
            ref["formalism"]
        for ref in revised_reference
    }

    for sample in samples:

        interpretation_id = (
            sample["interpretation_id"]
        )

        if (
            interpretation_id
            not in reference_by_id
        ):
            raise ValueError(
                "No revised gold reference for: "
                f"{interpretation_id}"
            )

        sample["gold_amr"] = (
            reference_by_id[
                interpretation_id
            ]
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated AMRs."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "JSON file containing generated AMRs."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output JSON path."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load generated AMRs
    # --------------------------------------------------------

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:

        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError(
            "Input JSON must be a list."
        )

    # --------------------------------------------------------
    # Load revised gold references
    # --------------------------------------------------------

    with AUGMENTED_JSONS.open(
        "r",
        encoding="utf-8",
    ) as f:

        revised_reference = json.load(f)

    if not isinstance(
        revised_reference,
        list,
    ):
        raise ValueError(
            "Revised reference JSON "
            "must be a list."
        )

    # --------------------------------------------------------
    # Replace old gold AMRs
    # --------------------------------------------------------

    replace_gold_references(
        samples,
        revised_reference,
    )

    # --------------------------------------------------------
    # Evaluate every sample
    # --------------------------------------------------------

    evaluated_samples = [
        evaluate_sample(sample)
        for sample in tqdm(
            samples,
            desc="Evaluating",
        )
    ]

    # --------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------

    summary = aggregate(
        evaluated_samples
    )

    invalid_reasons = (
        count_invalid_reasons(
            evaluated_samples
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_summary(
        summary
    )

    print_invalid_reasons(
        invalid_reasons
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "summary": summary,
        "invalid_reasons": invalid_reasons,
        "samples": evaluated_samples,
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

    print()
    print(
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()