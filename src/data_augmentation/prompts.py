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


def build_formalism_augmentation_prompt(sample: Dict[str, Any]) -> List[str]:
    # Prompt for formalism-based data augmentation
    # input: ambiguous caption, and interpretations based on which the formalism semantics is determined
    # output: a list of prompts for each interpretation
    
    ambiguous_caption = sample["ambiguous_caption"] 
    interpretations = sample["interpretations"] 

    prompts: List[str] = []
    
    for interpretation in interpretations:
        lines = [
            (
                "Your job is to generate an S-expression formalism for a "
                "structurally ambiguous caption."
            ),
            (
                "You will be provided with an ambiguous caption and a "
                "clarifying interpretation."
            ),
            (
                "Because of structural ambiguity, the caption can correspond "
                "to multiple syntactic-semantic structures. The clarifying "
                "interpretation specifies which structure is intended."
            ),
            (
                "Generate the S-expression formalism corresponding to the "
                "intended interpretation while preserving the lexical content "
                "of the original ambiguous caption."
            ),
            "",
            "Example:",
            "Ambiguous caption: \"The man saw the girl with a telescope.\"",
            (
                "Interpretation: "
                "\"The man used a telescope to see the girl.\""
            ),
            (
                "S-expression formalism: "
                "(S (NP The man) "
                "(VP saw (NP the girl) (PP with (NP a telescope))))"
            ),
            "",
            "Now process the following example:",
            f'Ambiguous caption: "{ambiguous_caption}"',
            f'Interpretation: "{interpretation}"',
            "",
            (
                "Return only the S-expression as a single string. "
                "Do not provide explanations, Markdown, or additional text."
            ),
        ]
        
        prompt = "\n".join(lines)
        prompts.append(prompt)
        
    return prompts
    
def build_realimage_based_caption_augmentation_prompts() -> Dict[str, str]:
    # Prompt to augment realimages with captions dealing with as many as structural ambiguities as possible 
    # Input: a real image (mostly from JCRE3)
    # Output: A bunch of ambiguous caption with clarifying interpretation affiliated with either of 7 categories defined by LaViSA
    
    prompts: List[str] = [] 
    
    for category, info in AMBIGUITY_CATEGORIES.items():
        lines = [
            (
                "Your job is to generate a structurally ambiguous caption "
                "for the provided real image."
            ),
            "",
            f"Target ambiguity category: {category}",
            f"Category name: {info['name']}",
            f"Definition: {info['description']}",
            "",
            "Example:",
            f"Ambiguous caption: {info['example_caption']}",
            f"Interpretation: {info['example_interpretation']}",
            "",
            (
                "Inspect the provided image and determine whether a natural "
                "caption exhibiting this specific type of structural "
                "ambiguity can be constructed from the visible scene."
            ),
            "",
            "Requirements:",
            (
                "- The ambiguous caption must have at least two plausible "
                "structural interpretations."
            ),
            (
                "- The ambiguity must arise specifically from the target "
                "category."
            ),
            (
                "- The clarifying interpretation must describe the "
                "interpretation supported by the provided image."
            ),
            (
                "- Do not invent objects, people, attributes, or actions "
                "that are not reasonably supported by the image."
            ),
            (
                "- The caption should sound like a natural description "
                "of the image."
            ),
            (
                "- If the image cannot naturally support this ambiguity "
                "category, do not force an example."
            ),
            "",
            "Return only valid JSON in one of the following forms:",
            "",
            "{",
            '  "valid": true,',
            f'  "category": "{category}",',
            '  "ambiguous_caption": "...",',
            '  "interpretation": "..."',
            "}",
            "",
            "or",
            "",
            "{",
            '  "valid": false,',
            f'  "category": "{category}",',
            '  "ambiguous_caption": null,',
            '  "interpretation": null',
            "}",
        ]
        
        prompt = "\n".join(lines)
        prompts.append(prompt)
    
    return prompts