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
    
    