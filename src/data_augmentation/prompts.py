from __future__ import annotations 

from typing import Any, Dict, List 


def build_formalism_augmentation_prompt(sample: Dict[str, Any]) -> List[str]:
    # Prompt for formalism-based data augmentation
    # input: ambiguous caption, and interpretations based on which the formalism semantics is determined
    # output: a list of prompts for each interpretation
    
    ambiguous_caption = sample["ambiguous_caption"] 
    interpretations = sample["interpretations"] 

    prompts: List[str] = []
    
    for interpretation in interpretations:
        lines = [
            "Your job is to generate an S-expression formalism for a given structurally ambiguous caption."
            "You will be provided with an ambiguous caption and a claryfing interpretation.",
            "Due to structural ambiguity, the given ambiguous caption can be parsed into multiple S-expressions, and a given interpretation will help you determine the correct S-expression formalism.",
            "So, your task is to generate the S-expression formalism for the given ambiguous caption, which reflects the meaning of the given interpretation.",
            "Please ensure that the generated S-expression formalism is syntactically correct and accurately represents the intended meaning of the ambiguous caption based on the provided interpretation.",
            "",
            "Example:",
            "Ambiguous caption: 'The man saw the girl with a telescope.'",
            "Interpretation: 'The man used a telescope to see the girl.'",
            "S-expression formalism: (S (NP The man) (VP saw (NP the girl) (PP with (NP a telescope))))",
            "",
            "Now, please generate the S-expression formalism for the following ambiguous caption and interpretation:",
            f'Ambiguous caption: "{ambiguous_caption}"',
            f'Interpretation: "{interpretation}"', 
            "",
            "Please provide your answer as a string of S-expression formalism without any additional text or explanation.",
        ]
        
        prompt = "\n".join(lines)
        prompts.append(prompt)
        
    return prompts
    
def build_realimage_based_caption_augmentation_prompt(None) -> List[str]:
    # Prompt to augment realimages with captions dealing with as many as structural ambiguities as possible 
    # Input: a real image (mostly from JCRE3)
    # Output: A bunch of ambiguous caption with clarifying interpretation affiliated with either of 7 categories defined by LaViSA
    
    lines = [
        "Your job is to generate structurally ambiguous captions for a given real image, along with clarifying interpretations that resolve the ambiguity.",
        "You will be provided with a real image, and your task is to think of a structurally ambiguous caption, and the interpretataion that could be derived from the given image.",
        "There are seven categories of structural ambiguity defined by our research, and I want you to generate captions associated with each category as possible.",
        "",
        "Here are the seven categories of structural ambiguity:",
        "1. Prepositional Phrase Attachment Ambiguity (pp): Ambiguity arises when a prepositional phrase can attach to different parts of the sentence, leading to multiple interpretations.",
        "e.g. ambiguous sentence: the man saw the girl with a telescope. interpretation: the man has the telescope and saw the girl"
        "2. Verb Phrase Attachment Ambiguity (vp): Ambiguity arises when a verb phrase can attach to different parts of the sentence, leading to multiple interpretations.",
        "e.g. ambiguous sentence: the man saw the girl approaching the house. interpretation: the man saw the girl while she was approaching the house", 
        "3. Anaphora Ambiguity (anaph): Ambiguity arises when a pronoun or noun phrase can refer to multiple antecedents, leading to multiple interpretations.",
        "e.g. ambiguous sentence: the man saw the girl and the woman, she was blonde. interpretation: the woman was blonde, not the girl",
        "4. Ellipsis Ambiguity (ellip): Ambiguity arises when a part of the sentence is omitted, leading to multiple interpretations.",
        "e.g. ambiguous sentence: the lion chased the tiger, also the bear. interpretation: the lion chased the tiger, and also chased the bear",
        "5. Adjective Scope Ambiguity (adj): Ambiguity arises when an adjective can modify different parts of the sentence, leading to multiple interpretations.",
        "e.g. ambiguous sentence: the yellow bag and chair. interpretation: the bag is yellow, but the chair is not",
        "6. Verb Scope Ambiguity (vb): Ambiguity arises when a verb can have different scopes, leading to multiple interpretations.",
        "e.g. ambiguous sentence: An elephant and a bird flying. interpretation: the elephant is flying as well as the bird",
        "7. Conjunction Ambiguity (conj): Ambiguity arises when a conjunction can connect different parts of the sentence, leading to multiple interpretations.",
        "e.g. ambiguous sentence: the man saw the girl and the boy or the woman. interpretation: the man saw the girl and the boy, but not the woman",
        "",
        "Now, please generate structurally ambiguous captions for the given real image, along with clarifying interpretations that resolve the ambiguity. Please ensure that the generated captions and interpretations are syntactically correct and accurately represent the intended meaning of the image based on the provided categories of structural ambiguity.",
        "Please provide your answer as a list of tuples, where each tuple contains a structurally ambiguous caption and its corresponding clarifying interpretation. Each tuple should be formatted as follows: (ambiguous caption, clarifying interpretation). Please do not include any additional text or explanation.",
    ]