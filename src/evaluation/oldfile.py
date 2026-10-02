from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import penman


PROP_BANK_RE = re.compile(r"^(.+)-(\d\d)$")
ARG_ROLE_RE = re.compile(r"^:ARG\d+$")

AMR_VARIABLE_DEF_RE = re.compile(
    r"\(\s*([A-Za-z][A-Za-z0-9_-]*)\s*/"
)


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def safe_prf(
    matched: int,
    predicted: int,
    gold: int,
) -> PRF:

    precision = (
        matched / predicted
        if predicted > 0
        else 0.0
    )

    recall = (
        matched / gold
        if gold > 0
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return PRF(
        precision=precision,
        recall=recall,
        f1=f1,
    )


# ============================================================
# Validity
# ============================================================

def find_duplicate_variables(
    amr: str,
) -> list[str]:

    variables = AMR_VARIABLE_DEF_RE.findall(
        amr
    )

    seen = set()
    duplicates = set()

    for variable in variables:

        if variable in seen:
            duplicates.add(
                variable
            )
        else:
            seen.add(
                variable
            )

    return sorted(
        duplicates
    )


def find_none_instances(
    graph: penman.Graph,
) -> list[str]:

    invalid = []

    for source, role, target in graph.instances():

        if target is None:
            invalid.append(
                source
            )

    return invalid


# ============================================================
# WSD
# ============================================================

def strip_wsd(
    concept: str | None,
) -> str | None:

    if not isinstance(
        concept,
        str,
    ):
        return concept

    match = PROP_BANK_RE.match(
        concept
    )

    if match is None:
        return concept

    return match.group(1)


def normalize_graph_wsd(
    graph: penman.Graph,
) -> penman.Graph:

    triples = []

    for source, role, target in graph.triples:

        if (
            role == ":instance"
            and isinstance(
                target,
                str,
            )
        ):
            target = strip_wsd(
                target
            )

        triples.append(
            (
                source,
                role,
                target,
            )
        )

    return penman.Graph(
        triples,
        top=graph.top,
    )


def encode_graph(
    graph: penman.Graph,
) -> str:

    return penman.encode(
        graph
    )


# ============================================================
# Concept F1
# ============================================================

def get_concepts(
    graph: penman.Graph,
    remove_wsd: bool = False,
) -> list[str]:

    concepts = []

    for source, role, target in graph.instances():

        if target is None:
            continue

        concept = target

        if remove_wsd:
            concept = strip_wsd(
                concept
            )

        if concept is not None:
            concepts.append(
                concept
            )

    return concepts


def compute_concept_f1_from_graphs(
    gold_graph: penman.Graph,
    pred_graph: penman.Graph,
    remove_wsd: bool = False,
) -> PRF:

    gold_concepts = get_concepts(
        gold_graph,
        remove_wsd=remove_wsd,
    )

    pred_concepts = get_concepts(
        pred_graph,
        remove_wsd=remove_wsd,
    )

    gold_counter = Counter(
        gold_concepts
    )

    pred_counter = Counter(
        pred_concepts
    )

    matched = sum(
        (
            gold_counter
            & pred_counter
        ).values()
    )

    return safe_prf(
        matched=matched,
        predicted=len(
            pred_concepts
        ),
        gold=len(
            gold_concepts
        ),
    )


# ============================================================
# SRL
# ============================================================

def get_concept_map(
    graph: penman.Graph,
    remove_wsd: bool = False,
) -> dict[str, str]:

    concept_map = {}

    for source, role, target in graph.instances():

        if target is None:
            continue

        concept = target

        if remove_wsd:
            concept = strip_wsd(
                concept
            )

        if concept is not None:
            concept_map[
                source
            ] = concept

    return concept_map

def get_coordination_members(
    graph: penman.Graph,
    variable: str,
    concept_map: dict[str, str],
) -> list[str]:
    """
    If variable is an 'and' / 'or' coordination node,
    return its coordinated members.

    Example:

        (a / and
            :op1 (b / book)
            :op2 (p / pen))

    returns:

        ["b", "p"]

    If it is not a coordination node, simply return
    the original variable.

    Nested coordination is expanded recursively.
    """

    concept = concept_map.get(
        variable
    )

    if concept not in {
        "and",
        "or",
    }:
        return [
            variable
        ]

    members = []

    for source, role, target in graph.edges():

        if source != variable:
            continue

        if not re.fullmatch(
            r":op\d+",
            role,
        ):
            continue

        members.extend(
            get_coordination_members(
                graph=graph,
                variable=target,
                concept_map=concept_map,
            )
        )

    # Safety fallback
    if not members:
        return [
            variable
        ]

    return members


def get_srl_triples(
    graph: penman.Graph,
    labeled: bool = True,
    remove_wsd: bool = False,
) -> list[tuple[str, str, str]]:

    concept_map = get_concept_map(
        graph,
        remove_wsd=remove_wsd,
    )

    triples = []

    for source, role, target in graph.edges():

        # -------------------------------
        # ARGx-of
        # -------------------------------

        if role.endswith(
            "-of"
        ):

            base_role = role[:-3]

            if not ARG_ROLE_RE.fullmatch(
                base_role
            ):
                continue

            predicate_var = target
            argument_var = source
            semantic_role = base_role

        # -------------------------------
        # ARGx
        # -------------------------------

        else:

            if not ARG_ROLE_RE.fullmatch(
                role
            ):
                continue

            predicate_var = source
            argument_var = target
            semantic_role = role

        predicate = concept_map.get(
            predicate_var
        )

        if predicate is None:
            continue

        # ------------------------------------------------
        # Expand coordinated arguments.
        #
        # Example:
        #
        # hold-01 :ARG1 and(book, pen)
        #
        # becomes:
        #
        # hold-01 :ARG1 book
        # hold-01 :ARG1 pen
        # ------------------------------------------------

        argument_vars = get_coordination_members(
            graph=graph,
            variable=argument_var,
            concept_map=concept_map,
        )

        for expanded_argument_var in argument_vars:

            argument = concept_map.get(
                expanded_argument_var
            )

            if argument is None:
                continue

            output_role = (
                semantic_role
                if labeled
                else ":ARG"
            )

            triples.append(
                (
                    predicate,
                    output_role,
                    argument,
                )
            )

    return triples


def compute_srl_f1_from_graphs(
    gold_graph: penman.Graph,
    pred_graph: penman.Graph,
    labeled: bool = True,
    remove_wsd: bool = False,
) -> PRF:

    gold_triples = get_srl_triples(
        gold_graph,
        labeled=labeled,
        remove_wsd=remove_wsd,
    )

    pred_triples = get_srl_triples(
        pred_graph,
        labeled=labeled,
        remove_wsd=remove_wsd,
    )

    gold_counter = Counter(
        gold_triples
    )

    pred_counter = Counter(
        pred_triples
    )

    matched = sum(
        (
            gold_counter
            & pred_counter
        ).values()
    )

    return safe_prf(
        matched=matched,
        predicted=len(
            pred_triples
        ),
        gold=len(
            gold_triples
        ),
    )