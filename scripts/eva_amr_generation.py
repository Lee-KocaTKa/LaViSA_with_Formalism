from __future__ import annotations

import argparse
import io
import json
import amr 
from collections import defaultdict
from pathlib import Path

import penman
import smatch
from tqdm import tqdm

from src.evaluation.amr_generation_evaluation import (
    compute_concept_f1_from_graphs,
    compute_srl_f1_from_graphs,
    encode_graph,
    find_duplicate_variables,
    find_none_instances,
    normalize_graph_wsd,
)


LIGHT_METRICS = [
    "concept_f1",
    "srl_smatch",
    "srl_smatch_no_wsd",
    "unlabeled_srl_smatch",
    "unlabeled_srl_smatch_no_wsd",
]


# ============================================================
# Utilities
# ============================================================

def is_smatch_parseable(amr_string: str) -> bool:
    try:
        parsed = amr.AMR.parse_AMR_line(
            amr_string.replace("\n", " ")
        )
        return parsed is not None
    except Exception:
        return False


def get_category(
    interpretation_id: str,
) -> str:

    return interpretation_id.split(
        "-",
        1,
    )[0]


def mean(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return sum(
        values
    ) / len(
        values
    )


# ============================================================
# Smatch
# ============================================================

def build_smatch_stream(
    amrs: list[str],
) -> io.StringIO:

    text = "\n\n".join(
        amr.strip()
        for amr in amrs
    )

    text += "\n\n"

    return io.StringIO(
        text
    )


def compute_corpus_smatch(
    gold_amrs: list[str],
    pred_amrs: list[str],
) -> float:

    if len(
        gold_amrs
    ) != len(
        pred_amrs
    ):
        raise ValueError(
            f"Gold/pred mismatch: "
            f"{len(gold_amrs)} vs "
            f"{len(pred_amrs)}"
        )

    if not gold_amrs:
        return 0.0

    gold_stream = build_smatch_stream(
        gold_amrs
    )

    pred_stream = build_smatch_stream(
        pred_amrs
    )

    results = list(
        smatch.score_amr_pairs(
            gold_stream,
            pred_stream,
        )
    )

    if not results:
        return 0.0

    precision, recall, f1 = (
        results[-1]
    )

    return f1


# ============================================================
# Preparation
# ============================================================

def prepare_samples(
    samples: list[dict],
):
    evaluated_samples = []

    smatch_data = {
        "overall": {
            "gold": [],
            "pred": [],
            "gold_no_wsd": [],
            "pred_no_wsd": [],
        },
        "by_category": defaultdict(
            lambda: {
                "gold": [],
                "pred": [],
                "gold_no_wsd": [],
                "pred_no_wsd": [],
            }
        ),
    }

    validation_stats = {
        "total_samples": len(
            samples
        ),

        "generation_error": 0,
        "missing_gold_amr": 0,
        "missing_generated_amr": 0,

        "generated_parse_failure": 0,
        "generated_parse_valid": 0,

        "generated_duplicate_variable": 0,
        "generated_none_instance": 0,

        "generated_strict_valid": 0,

        "semantic_evaluable_samples": 0,
        "smatch_evaluable_samples": 0,

        "by_category": {},
    }

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "parse_valid": 0,
            "strict_valid": 0,
            "semantic_evaluable": 0,
            "smatch_evaluable": 0,
            "duplicate_variable": 0,
            "none_instance": 0,
        }
    )

    print(
        "Parsing and validating AMRs..."
    )

    for original_sample in tqdm(
        samples,
        desc="Parsing",
    ):

        sample = dict(
            original_sample
        )

        interpretation_id = sample.get(
            "interpretation_id",
            "unknown",
        )

        category = get_category(
            interpretation_id
        )

        sample[
            "category"
        ] = category

        category_stats[
            category
        ][
            "total"
        ] += 1

        sample["metrics"] = {
            "concept_f1": None,
            "srl_smatch": None,
            "srl_smatch_no_wsd": None,
            "unlabeled_srl_smatch": None,
            "unlabeled_srl_smatch_no_wsd": None,
        }

        gold_amr = sample.get(
            "gold_amr"
        )

        pred_amr = sample.get(
            "generated_amr"
        )

        existing_error = sample.get(
            "error"
        )

        # ----------------------------------------------------
        # Generation error
        # ----------------------------------------------------

        if existing_error is not None:

            validation_stats[
                "generation_error"
            ] += 1

            sample[
                "evaluation_error"
            ] = (
                f"Generation error: "
                f"{existing_error}"
            )

            evaluated_samples.append(
                sample
            )

            continue

        # ----------------------------------------------------
        # Missing AMRs
        # ----------------------------------------------------

        if not gold_amr:

            validation_stats[
                "missing_gold_amr"
            ] += 1

            sample[
                "evaluation_error"
            ] = (
                "Missing gold AMR"
            )

            evaluated_samples.append(
                sample
            )

            continue

        if not pred_amr:

            validation_stats[
                "missing_generated_amr"
            ] += 1

            sample[
                "evaluation_error"
            ] = (
                "Missing generated AMR"
            )

            evaluated_samples.append(
                sample
            )

            continue

        # ----------------------------------------------------
        # Duplicate variables
        # ----------------------------------------------------

        duplicate_variables = (
            find_duplicate_variables(
                pred_amr
            )
        )

        if duplicate_variables:

            validation_stats[
                "generated_duplicate_variable"
            ] += 1

            category_stats[
                category
            ][
                "duplicate_variable"
            ] += 1

        # ----------------------------------------------------
        # Parse
        # ----------------------------------------------------

        try:

            gold_graph = penman.decode(
                gold_amr
            )

            pred_graph = penman.decode(
                pred_amr
            )

        except Exception as exc:

            validation_stats[
                "generated_parse_failure"
            ] += 1

            sample[
                "evaluation_error"
            ] = (
                "PENMAN parse error: "
                f"{exc}"
            )

            evaluated_samples.append(
                sample
            )

            continue

        validation_stats[
            "generated_parse_valid"
        ] += 1

        category_stats[
            category
        ][
            "parse_valid"
        ] += 1

        # ----------------------------------------------------
        # None instances
        # ----------------------------------------------------

        none_instances = (
            find_none_instances(
                pred_graph
            )
        )

        if none_instances:

            validation_stats[
                "generated_none_instance"
            ] += 1

            category_stats[
                category
            ][
                "none_instance"
            ] += 1

        strict_valid = (
            len(
                duplicate_variables
            ) == 0
            and len(
                none_instances
            ) == 0
        )

        if strict_valid:

            validation_stats[
                "generated_strict_valid"
            ] += 1

            category_stats[
                category
            ][
                "strict_valid"
            ] += 1

        sample[
            "amr_validation"
        ] = {
            "duplicate_variables":
                duplicate_variables,

            "none_instance_variables":
                none_instances,

            "strict_valid":
                strict_valid,
        }

        # ----------------------------------------------------
        # Concept + SRL metrics
        # ----------------------------------------------------

        try:

            concept = (
                compute_concept_f1_from_graphs(
                    gold_graph,
                    pred_graph,
                )
            )

            srl = (
                compute_srl_f1_from_graphs(
                    gold_graph,
                    pred_graph,
                    labeled=True,
                )
            )

            unlabeled_srl = (
                compute_srl_f1_from_graphs(
                    gold_graph,
                    pred_graph,
                    labeled=False,
                )
            )
            
            srl_no_wsd = compute_srl_f1_from_graphs(
                gold_graph,
                pred_graph,
                labeled=True,
                remove_wsd=True,
            )

            unlabeled_srl_no_wsd = compute_srl_f1_from_graphs(
                gold_graph,
                pred_graph,
                labeled=False,
                remove_wsd=True,
            )

            sample[
                "metrics"
            ][
                "concept_f1"
            ] = concept.f1

            sample[
                "metrics"
            ][
                "srl_smatch"
            ] = srl.f1
            
            sample[
                "metrics"
            ][
                "srl_smatch_no_wsd"
                        ] = srl_no_wsd.f1
            

            sample[
                "metrics"
            ][
                "unlabeled_srl_smatch"
            ] = unlabeled_srl.f1
            
            sample["metrics"]["unlabeled_srl_smatch_no_wsd"] = unlabeled_srl_no_wsd.f1 

            validation_stats[
                "semantic_evaluable_samples"
            ] += 1

            category_stats[
                category
            ][
                "semantic_evaluable"
            ] += 1

        except Exception as exc:

            sample[
                "evaluation_error"
            ] = (
                "Concept/SRL error: "
                f"{exc}"
            )

            evaluated_samples.append(
                sample
            )

            continue

        # ----------------------------------------------------
        # Prepare Smatch representations
        # ----------------------------------------------------

        try:

            encoded_gold = (
                encode_graph(
                    gold_graph
                )
            )

            encoded_pred = (
                encode_graph(
                    pred_graph
                )
            )

            gold_no_wsd = (
                encode_graph(
                    normalize_graph_wsd(
                        gold_graph
                    )
                )
            )

            pred_no_wsd = (
                encode_graph(
                    normalize_graph_wsd(
                        pred_graph
                    )
                )
            )

        except Exception as exc:

            sample[
                "evaluation_error"
            ] = (
                "Encoding error: "
                f"{exc}"
            )

            evaluated_samples.append(
                sample
            )

            continue

        smatch_ok = (
            is_smatch_parseable(encoded_gold) 
            and is_smatch_parseable(encoded_pred)
            and is_smatch_parseable(gold_no_wsd)
            and is_smatch_parseable(pred_no_wsd)
        )

        # ----------------------------------------------------
        # Overall Smatch corpus
        # ----------------------------------------------------

        if smatch_ok:
            
            smatch_data[
                "overall"
            ][
                "gold"
            ].append(
                encoded_gold
            )

            smatch_data[
                "overall"
            ][
                "pred"
            ].append(
                encoded_pred
            )

            smatch_data[
                "overall"
            ][
                "gold_no_wsd"
            ].append(
                gold_no_wsd
            )

            smatch_data[
                "overall"
            ][
                "pred_no_wsd"
            ].append(
                pred_no_wsd
            )

        # ----------------------------------------------------
        # Category Smatch corpus
        # ----------------------------------------------------

            category_data = (
                smatch_data[
                    "by_category"
                ][
                    category
                ]
            )

            category_data[
                "gold"
            ].append(
                encoded_gold
            )

            category_data[
                "pred"
            ].append(
                encoded_pred
            )

            category_data[
                "gold_no_wsd"
            ].append(
                gold_no_wsd
            )

            category_data[
                "pred_no_wsd"
            ].append(
                pred_no_wsd
            )

            validation_stats[
                "smatch_evaluable_samples"
            ] += 1

            category_stats[
                category
            ][
                "smatch_evaluable"
            ] += 1

            evaluated_samples.append(
                sample
            )
            
        else:
            sample["smatch_parse_valid"] = False
            
        sample["smatch_parse_valid"] = smatch_ok 

    # ========================================================
    # Rates
    # ========================================================

    total = validation_stats[
        "total_samples"
    ]

    if total:

        validation_stats[
            "parse_valid_rate"
        ] = (
            validation_stats[
                "generated_parse_valid"
            ]
            / total
        )

        validation_stats[
            "strict_valid_rate"
        ] = (
            validation_stats[
                "generated_strict_valid"
            ]
            / total
        )

        validation_stats[
            "semantic_evaluable_rate"
        ] = (
            validation_stats[
                "semantic_evaluable_samples"
            ]
            / total
        )

        validation_stats[
            "smatch_evaluable_rate"
        ] = (
            validation_stats[
                "smatch_evaluable_samples"
            ]
            / total
        )

    else:

        validation_stats[
            "parse_valid_rate"
        ] = 0.0

        validation_stats[
            "strict_valid_rate"
        ] = 0.0

        validation_stats[
            "semantic_evaluable_rate"
        ] = 0.0

        validation_stats[
            "smatch_evaluable_rate"
        ] = 0.0

    # ========================================================
    # Category validity
    # ========================================================

    for category in sorted(
        category_stats
    ):

        stats = category_stats[
            category
        ]

        category_total = stats[
            "total"
        ]

        if category_total:

            stats[
                "parse_valid_rate"
            ] = (
                stats[
                    "parse_valid"
                ]
                / category_total
            )

            stats[
                "strict_valid_rate"
            ] = (
                stats[
                    "strict_valid"
                ]
                / category_total
            )

            stats[
                "semantic_evaluable_rate"
            ] = (
                stats[
                    "semantic_evaluable"
                ]
                / category_total
            )

            stats[
                "smatch_evaluable_rate"
            ] = (
                stats[
                    "smatch_evaluable"
                ]
                / category_total
            )

        else:

            stats[
                "parse_valid_rate"
            ] = 0.0

            stats[
                "strict_valid_rate"
            ] = 0.0

            stats[
                "semantic_evaluable_rate"
            ] = 0.0

            stats[
                "smatch_evaluable_rate"
            ] = 0.0

        validation_stats[
            "by_category"
        ][
            category
        ] = dict(
            stats
        )

    return (
        evaluated_samples,
        smatch_data,
        validation_stats,
    )


# ============================================================
# Lightweight metric aggregation
# ============================================================

def aggregate_light_metrics(
    samples: list[dict],
) -> dict:

    overall = defaultdict(
        list
    )

    by_category = defaultdict(
        lambda: defaultdict(
            list
        )
    )

    for sample in samples:

        metrics = sample.get(
            "metrics"
        )

        if not metrics:
            continue

        category = sample.get(
            "category"
        )

        for metric_name in LIGHT_METRICS:

            score = metrics.get(
                metric_name
            )

            if score is None:
                continue

            overall[
                metric_name
            ].append(
                score
            )

            by_category[
                category
            ][
                metric_name
            ].append(
                score
            )

    result = {
        "overall": {},
        "by_category": {},
    }

    for metric_name in LIGHT_METRICS:

        values = overall.get(
            metric_name,
            [],
        )

        result[
            "overall"
        ][
            metric_name
        ] = {
            "score": mean(
                values
            ),
            "count": len(
                values
            ),
        }

    for category in sorted(
        by_category
    ):

        result[
            "by_category"
        ][
            category
        ] = {}

        for metric_name in LIGHT_METRICS:

            values = (
                by_category[
                    category
                ].get(
                    metric_name,
                    [],
                )
            )

            result[
                "by_category"
            ][
                category
            ][
                metric_name
            ] = {
                "score": mean(
                    values
                ),
                "count": len(
                    values
                ),
            }

    return result


# ============================================================
# Smatch aggregation
# ============================================================

def compute_all_smatch_scores(
    smatch_data: dict,
) -> dict:

    result = {
        "overall": {},
        "by_category": {},
    }

    overall = smatch_data[
        "overall"
    ]

    print()
    print(
        f"Computing overall Smatch "
        f"({len(overall['gold'])} samples)..."
    )

    result[
        "overall"
    ][
        "smatch"
    ] = {
        "score": compute_corpus_smatch(
            overall["gold"],
            overall["pred"],
        ),
        "count": len(
            overall["gold"]
        ),
    }

    print(
        "Computing overall "
        "Smatch No-WSD..."
    )

    result[
        "overall"
    ][
        "smatch_no_wsd"
    ] = {
        "score": compute_corpus_smatch(
            overall[
                "gold_no_wsd"
            ],
            overall[
                "pred_no_wsd"
            ],
        ),
        "count": len(
            overall[
                "gold_no_wsd"
            ]
        ),
    }

    for category, data in sorted(
        smatch_data[
            "by_category"
        ].items()
    ):

        print()
        print(
            f"Computing {category} Smatch "
            f"({len(data['gold'])} samples)..."
        )

        smatch_score = (
            compute_corpus_smatch(
                data["gold"],
                data["pred"],
            )
        )

        print(
            f"Computing {category} "
            f"Smatch No-WSD..."
        )

        no_wsd_score = (
            compute_corpus_smatch(
                data[
                    "gold_no_wsd"
                ],
                data[
                    "pred_no_wsd"
                ],
            )
        )

        result[
            "by_category"
        ][
            category
        ] = {
            "smatch": {
                "score":
                    smatch_score,
                "count":
                    len(
                        data["gold"]
                    ),
            },

            "smatch_no_wsd": {
                "score":
                    no_wsd_score,
                "count":
                    len(
                        data[
                            "gold_no_wsd"
                        ]
                    ),
            },
        }

    return result


# ============================================================
# Merge summaries
# ============================================================

def merge_summaries(
    light_metrics: dict,
    smatch_metrics: dict,
) -> dict:

    summary = {
        "overall": {},
        "by_category": {},
    }

    summary[
        "overall"
    ][
        "smatch"
    ] = smatch_metrics[
        "overall"
    ][
        "smatch"
    ]

    summary[
        "overall"
    ][
        "smatch_no_wsd"
    ] = smatch_metrics[
        "overall"
    ][
        "smatch_no_wsd"
    ]

    for metric_name in LIGHT_METRICS:

        summary[
            "overall"
        ][
            metric_name
        ] = light_metrics[
            "overall"
        ][
            metric_name
        ]

    categories = sorted(
        set(
            light_metrics[
                "by_category"
            ].keys()
        )
        |
        set(
            smatch_metrics[
                "by_category"
            ].keys()
        )
    )

    for category in categories:

        summary[
            "by_category"
        ][
            category
        ] = {}

        smatch_category = (
            smatch_metrics[
                "by_category"
            ].get(
                category,
                {},
            )
        )

        light_category = (
            light_metrics[
                "by_category"
            ].get(
                category,
                {},
            )
        )

        summary[
            "by_category"
        ][
            category
        ][
            "smatch"
        ] = smatch_category.get(
            "smatch",
            {
                "score": 0.0,
                "count": 0,
            },
        )

        summary[
            "by_category"
        ][
            category
        ][
            "smatch_no_wsd"
        ] = smatch_category.get(
            "smatch_no_wsd",
            {
                "score": 0.0,
                "count": 0,
            },
        )

        for metric_name in LIGHT_METRICS:

            summary[
                "by_category"
            ][
                category
            ][
                metric_name
            ] = light_category.get(
                metric_name,
                {
                    "score": 0.0,
                    "count": 0,
                },
            )

    return summary


# ============================================================
# Printing
# ============================================================

def print_validation_summary(
    stats: dict,
) -> None:

    print()
    print(
        "=" * 90
    )

    print(
        "AMR VALIDITY"
    )

    print(
        "=" * 90
    )

    print(
        f"Total samples:                 "
        f"{stats['total_samples']}"
    )

    print(
        f"Generation errors:             "
        f"{stats['generation_error']}"
    )

    print(
        f"Missing generated AMR:         "
        f"{stats['missing_generated_amr']}"
    )

    print(
        f"Generated parse failures:      "
        f"{stats['generated_parse_failure']}"
    )

    print(
        f"Duplicate-variable AMRs:       "
        f"{stats['generated_duplicate_variable']}"
    )

    print(
        f"None-instance AMRs:            "
        f"{stats['generated_none_instance']}"
    )

    print()

    print(
        f"Parse-valid AMRs:              "
        f"{stats['generated_parse_valid']} "
        f"({stats['parse_valid_rate'] * 100:.2f}%)"
    )

    print(
        f"Strict-valid AMRs:             "
        f"{stats['generated_strict_valid']} "
        f"({stats['strict_valid_rate'] * 100:.2f}%)"
    )

    print(
        f"Semantic evaluable:            "
        f"{stats['semantic_evaluable_samples']} "
        f"({stats['semantic_evaluable_rate'] * 100:.2f}%)"
    )

    print(
        f"Smatch evaluable:              "
        f"{stats['smatch_evaluable_samples']} "
        f"({stats['smatch_evaluable_rate'] * 100:.2f}%)"
    )

    print()
    print(
        "VALIDITY BY CATEGORY"
    )

    print(
        "-" * 90
    )

    print(
        f"{'Category':<10}"
        f"{'N':>7}"
        f"{'Parse':>13}"
        f"{'Strict':>13}"
        f"{'Semantic':>13}"
        f"{'Smatch':>13}"
    )

    print(
        "-" * 90
    )

    for category, data in (
        stats[
            "by_category"
        ].items()
    ):

        print(
            f"{category:<10}"
            f"{data['total']:>7}"
            f"{data['parse_valid_rate'] * 100:>12.2f}%"
            f"{data['strict_valid_rate'] * 100:>12.2f}%"
            f"{data['semantic_evaluable_rate'] * 100:>12.2f}%"
            f"{data['smatch_evaluable_rate'] * 100:>12.2f}%"
        )


def print_metric_summary(
    summary: dict,
) -> None:

    metric_order = [
        "smatch",
        "smatch_no_wsd",
        "concept_f1",
        "srl_smatch",
        "srl_smatch_no_wsd",
        "unlabeled_srl_smatch",
        "unlabeled_srl_smatch_no_wsd",
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "AMR GENERATION METRICS"
    )

    print(
        "=" * 100
    )

    print()
    print(
        "OVERALL"
    )

    print(
        "-" * 65
    )

    print(
        f"{'Metric':<32}"
        f"{'Score':>12}"
        f"{'N':>10}"
    )

    print(
        "-" * 65
    )

    for metric_name in metric_order:

        result = summary[
            "overall"
        ][
            metric_name
        ]

        print(
            f"{metric_name:<32}"
            f"{result['score'] * 100:>11.2f}%"
            f"{result['count']:>10}"
        )

    print()
    print(
        "BY CATEGORY"
    )

    print(
        "-" * 120
    )

    header = (
        f"{'Cat':<10}"
    )

    for metric_name in metric_order:

        header += (
            f"{metric_name:>22}"
        )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for category, results in (
        summary[
            "by_category"
        ].items()
    ):

        row = (
            f"{category:<10}"
        )

        for metric_name in metric_order:

            result = results.get(
                metric_name,
                {
                    "score": 0.0,
                    "count": 0,
                },
            )

            row += (
                f"{result['score'] * 100:>15.2f}%"
                f"({result['count']:>4})"
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
            "Evaluate generated AMRs."
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
            "Input JSON must be a list."
        )

    (
        evaluated_samples,
        smatch_data,
        validation_stats,
    ) = prepare_samples(
        samples
    )

    light_metrics = (
        aggregate_light_metrics(
            evaluated_samples
        )
    )

    smatch_metrics = (
        compute_all_smatch_scores(
            smatch_data
        )
    )

    summary = merge_summaries(
        light_metrics,
        smatch_metrics,
    )

    output = {
        "validation":
            validation_stats,

        "summary":
            summary,

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

    print_validation_summary(
        validation_stats
    )

    print_metric_summary(
        summary
    )

    print()
    print(
        f"Saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()