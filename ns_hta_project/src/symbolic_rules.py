"""
symbolic_rules.py - Extraction of 5 rule-derived binary symptom features.

Features:
s1: spots
s2: powdery growth
s3: leaf curling
s4: blight-related symptoms
s5: yellowing

Rule application note:
Rules are applied to sanitized text to prevent diagnostic label leakage from
bypassing the text encoder and entering through symbolic features.
"""

import re
import numpy as np
from typing import List, Dict, Tuple


# Rule specifications with regex patterns and descriptions
SYMBOLIC_RULES = {
    0: {
        "name": "spots",
        "description": "Localized spots, lesions, or speckling on leaves",
        "patterns": [
            r"\bspots?\b",
            r"\blesions?\b",
            r"\bspeckl(?:ing|es?)\b",
            r"\bcircular\s+(?:marks?|dots?|patches?)\b",
            r"\bnecrotic\s+spots?\b",
            r"\bdark\s+spots?\b",
            r"\bbrown\s+spots?\b"
        ]
    },
    1: {
        "name": "powdery_growth",
        "description": "White, talcum-like, or fluffy mycelial fungal growth",
        "patterns": [
            r"\bpowdery\b",
            r"\bwhite\s+(?:coating|dust|powder|fungal\s+growth|patches?|mould|mildew)\b",
            r"\btalcum\b",
            r"\bfluffy\s+growth\b",
            r"\bvelvety\s+layer\b"
        ]
    },
    2: {
        "name": "leaf_curling",
        "description": "Mechanical deformation, rolling, cupping, or curling",
        "patterns": [
            r"\bcurl(?:ing|ed|s)?\b",
            r"\broll(?:ing|ed|s)?\b",
            r"\bcrinkl(?:ing|ed|es)?\b",
            r"\bdistortion\b",
            r"\bcupping\b",
            r"\bwrinkled\b",
            r"\bdeformed\s+leaves?\b"
        ]
    },
    3: {
        "name": "blight_symptoms",
        "description": "Water-soaked lesions, rapid tissue collapse, necrosis",
        "patterns": [
            r"\bwater[- ]soaked\b",
            r"\bblight\b",
            r"\brapid\s+browning\b",
            r"\bnecrosis\b",
            r"\bdying\s+tissue\b",
            r"\bwilting\s+patches?\b",
            r"\bextensive\s+browning\b"
        ]
    },
    4: {
        "name": "yellowing",
        "description": "Chlorosis, yellow margins, yellow halos, pale foliage",
        "patterns": [
            r"\byellow(?:ing|ish|ed)?\b",
            r"\bchloros(?:is|tic)\b",
            r"\byellow\s+halo\b",
            r"\bpale\s+(?:green|yellow|foliage|leaves?)\b",
            r"\bloss\s+of\s+green\b"
        ]
    }
}


def extract_symbolic_features(text: str) -> np.ndarray:
    """
    Extracts a 5-dimensional binary symptom indicator vector x_S from a text description.
    
    Args:
        text (str): Input symptom description.
        
    Returns:
        np.ndarray of shape (5,) with binary values {0, 1}.
    """
    if not isinstance(text, str):
        return np.zeros(5, dtype=np.float32)
        
    text_lower = text.lower()
    vector = np.zeros(5, dtype=np.float32)
    
    for idx, rule in SYMBOLIC_RULES.items():
        matched = False
        for pattern in rule["patterns"]:
            if re.search(pattern, text_lower):
                matched = True
                break
        if matched:
            vector[idx] = 1.0
            
    return vector


def batch_extract_symbolic_features(texts: List[str]) -> np.ndarray:
    """
    Extracts 5-dimensional binary symbolic features for a list of texts.
    
    Returns:
        np.ndarray of shape (N, 5)
    """
    return np.array([extract_symbolic_features(t) for t in texts], dtype=np.float32)


def get_rule_descriptions_table() -> List[Dict]:
    """
    Returns a structured summary table of the symbolic rules for reporting in the paper.
    """
    table = []
    for idx, rule in SYMBOLIC_RULES.items():
        table.append({
            "Indicator": f"s_{idx+1}",
            "Symptom Category": rule["name"].replace("_", " ").title(),
            "Clinical Description": rule["description"],
            "Key Regex Patterns": ", ".join(rule["patterns"][:3]) + ", ..."
        })
    return table
