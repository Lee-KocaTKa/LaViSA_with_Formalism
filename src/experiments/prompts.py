from __future__ import annotations 

from typing import Any, Dict, List

AMBIGUITY_CATEGORIES = {
    "pp": {
        "name": "Prepositional Phrase Attachment Ambiguity",
        "description": (
            "Ambiguity arises when a prepositional phrase can attach to different "
            "parts of the sentence, leading to multiple interpretations."
        ),
        "example_caption": "The man saw the girl with a telescope.",        
    "example_interpretation": (
            "The man has the telescope and saw the girl."
    ),
    },
    "vp": {
        "name": "Verb Phrase Attachment Ambiguity",
        "description": (
            "Ambiguity arises when a verb phrase can attach to different "
            "parts of the sentence, leading to multiple interpretations."
        ),
        "example_caption": "The man saw the girl approaching the house.",
        "example_interpretation": (
            "The man saw the girl while she was approaching the house."
    ),  
    },
    "anaph": {
        "name": "Anaphora Ambiguity",
        "description": (
            "Ambiguity arises when a pronoun or noun phrase can refer to multiple "
            "antecedents, leading to multiple interpretations."
        ),
        "example_caption": "The man saw the girl and the woman, she was blonde.",
        "example_interpretation": (
            "The woman was blonde, not the girl."
        )
    },
    "ellip": {
        "name": "Ellipsis Ambiguity",
        "description": (
            "Ambiguity arises when a part of the sentence is omitted, leading to "
            "multiple interpretations."
        ),
        "example_caption": "The lion chased the tiger, also the bear.",
        "example_interpretation": (
            "The lion chased the tiger, and also chased the bear."
        )
    },
    "adj": {
        "name": "Adjective Scope Ambiguity",
        "description": (
            "Ambiguity arises when an adjective can modify different parts of the "
            "sentence, leading to multiple interpretations."
        ),
        "example_caption": "The yellow bag and chair.",
        "example_interpretation": (
            "The bag is yellow, but the chair is not."
        ),
    },
    "vb": {
        "name": "Verb Scope Ambiguity",
        "description": (
            "Ambiguity arises when a verb can have different scopes, leading to "
            "multiple interpretations."
        ),
        "example_caption": "An elephant and a bird flying.",
        "example_interpretation": (
            "The elephant is flying as well as the bird."
        ),
    },
    "conj": {
        "name": "Conjunction Ambiguity",
        "description": (
            "Ambiguity arises when a conjunction can connect different parts of the "
            "sentence, leading to multiple interpretations."
        ),
        "example_caption": "The man saw the girl and the boy or the woman.",
        "example_interpretation": (
            "The man saw the girl and the boy, but not the woman."
        ),  
    }
}

def build_amr_generation_prompt(
    sample: Dict[str, Any],
) -> str:
    """
    Task:
        ambiguous caption + clarifying image -> AMR

    The image itself is passed separately to the VLM.
    """

    ambiguous_caption = sample["ambiguous_caption"]

    lines = [
        (
            "Your job is to generate an Abstract Meaning Representation (AMR) "
            "for a structurally ambiguous caption."
        ),
        (
            "You will be provided with an ambiguous caption and a clarifying image."
        ),
        (
            "Because of structural ambiguity, the caption can correspond to "
            "multiple semantic structures. The image specifies which meaning "
            "is intended."
        ),
        (
            "Generate a standard AMR graph in PENMAN notation that represents "
            "the meaning of the ambiguous caption as disambiguated by the image."
        ),
        (
            "The AMR must reflect the semantic relations supported by the image, "
            "including predicate-argument relations, attachment, coreference, "
            "scope, or coordination when relevant."
        ),
        (
            "Use standard AMR conventions and PropBank-style predicate senses "
            "where appropriate."
        ),
        "",
        "Example:",
        'Ambiguous caption: "The man saw the girl with a telescope."',
        (
            "Suppose the image shows the man looking through a telescope "
            "at the girl."
        ),
        (
            "AMR: "
            "(s / see-01 "
            ":ARG0 (m / man) "
            ":ARG1 (g / girl) "
            ":instrument (t / telescope))"
        ),
        "",
        "Now process the following example:",
        f'Ambiguous caption: "{ambiguous_caption}"',
        "",
        (
            "Use the provided image to determine the intended interpretation."
        ),
        (
            "Return only the AMR graph in valid PENMAN notation as a single string. "
            "Do not provide explanations, Markdown, comments, or additional text."
        ),
    ]

    return "\n".join(lines)


def build_amr_verification_prompt(
    sample: Dict[str, Any],
) -> str:
    """
    Task:
        image + AMR -> Yes / No

    The image itself is passed separately to the VLM.
    """

    amr = sample["amr"]

    lines = [
        (
            "Your job is to determine whether an Abstract Meaning Representation "
            "(AMR) correctly corresponds to the provided image."
        ),
        (
            "The AMR describes a semantic interpretation involving entities, "
            "events, predicate-argument relations, attachment, coreference, "
            "scope, or coordination."
        ),
        (
            "Compare the semantic structure expressed by the AMR with the visual "
            "scene in the image."
        ),
        (
            "Answer Yes only if the image supports the semantic relations "
            "expressed by the AMR."
        ),
        (
            "Answer No if any semantically important relation in the AMR is "
            "inconsistent with the image."
        ),
        "",
        "AMR:",
        amr,
        "",
        (
            'Return only "Yes" or "No". '
            "Do not provide explanations, Markdown, comments, or additional text."
        ),
    ]

    return "\n".join(lines)


def build_i2t_matching_prompt(
    sample: Dict[str, Any],
) -> str:
    """
    Task:
        one image + two AMRs -> label of matching AMR

    The image itself is passed separately to the VLM.
    """

    amr_1 = sample["amr_1"]
    amr_2 = sample["amr_2"]

    lines = [
        (
            "Your job is to select which Abstract Meaning Representation (AMR) "
            "best corresponds to the provided image."
        ),
        (
            "The two AMRs represent different semantic interpretations of the "
            "same structurally ambiguous caption."
        ),
        (
            "Compare the visual scene with the semantic structure of each AMR, "
            "including predicate-argument relations, attachment, coreference, "
            "scope, and coordination when relevant."
        ),
        "",
        "Option 1:",
        amr_1,
        "",
        "Option 2:",
        amr_2,
        "",
        (
            "Select the AMR whose semantic interpretation is best supported "
            "by the image."
        ),
        (
            'Return only "1" or "2". '
            "Do not provide explanations, Markdown, comments, or additional text."
        ),
    ]

    return "\n".join(lines)


def build_t2i_matching_prompt(
    sample: Dict[str, Any],
) -> str:
    """
    Task:
        one AMR + two images -> label of matching image

    The two images must be passed separately to the VLM, in the same order
    as Image 1 and Image 2 in this prompt.
    """

    amr = sample["amr"]

    lines = [
        (
            "Your job is to select which of two provided images best corresponds "
            "to an Abstract Meaning Representation (AMR)."
        ),
        (
            "The AMR specifies a semantic interpretation involving entities, "
            "events, predicate-argument relations, attachment, coreference, "
            "scope, or coordination."
        ),
        (
            "Compare the semantic structure expressed by the AMR with both images."
        ),
        "",
        "AMR:",
        amr,
        "",
        (
            "The first provided image is Image 1."
        ),
        (
            "The second provided image is Image 2."
        ),
        "",
        (
            "Select the image whose visual scene best matches the semantic "
            "interpretation represented by the AMR."
        ),
        (
            'Return only "1" or "2". '
            "Do not provide explanations, Markdown, comments, or additional text."
        ),
    ]

    return "\n".join(lines)