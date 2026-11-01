"""
sanitization.py - Text sanitization and leakage audit for NS-HTA framework.

Implements:
1. Lexical sanitization function S(d_i) removing explicit disease names.
2. Leakage auditor detecting diagnostic label terms in raw vs. sanitized text.
3. Negation phrase analyzer explaining why Healthy records contain disease terms.
"""

import re
from typing import Dict, List, Tuple


# Regex patterns for explicit disease names and diagnostic labels
DISEASE_PATTERNS = [
    # Powdery Mildew
    (r"\bpowdery\s+mildew\b", "[DISEASE_REMOVED]"),
    (r"\bmildew\b", "[DISEASE_REMOVED]"),
    (r"\bpowdery\b", "[SYMPTOM_REMOVED]"),
    
    # Early Blight
    (r"\bearly\s+blight\b", "[DISEASE_REMOVED]"),
    (r"\balternaria\b", "[PATHOGEN_REMOVED]"),
    (r"\balternaria\s+solani\b", "[PATHOGEN_REMOVED]"),
    
    # Late Blight
    (r"\blate\s+blight\b", "[DISEASE_REMOVED]"),
    (r"\bphytophthora\b", "[PATHOGEN_REMOVED]"),
    (r"\bphytophthora\s+infestans\b", "[PATHOGEN_REMOVED]"),
    
    # Generic Blight
    (r"\bblight\b", "[DISEASE_REMOVED]"),
    
    # Healthy / Normal explicit label leaks
    (r"\bhealthy\s+plant\b", "[STATUS_REMOVED]"),
    (r"\bhealthy\s+crop\b", "[STATUS_REMOVED]"),
    (r"\bhealthy\s+leaf\b", "[STATUS_REMOVED]"),
    (r"\bhealthy\s+leaves\b", "[STATUS_REMOVED]"),
    (r"\bno\s+disease\b", "[STATUS_REMOVED]"),
    (r"\bdisease[- ]free\b", "[STATUS_REMOVED]"),
]

# Diagnostic terms used for leakage audit
AUDIT_KEYWORDS = {
    "Healthy": [
        "healthy", "normal", "no disease", "disease-free", "unaffected", "free of disease", "healthy leaf"
    ],
    "Powdery Mildew": [
        "powdery mildew", "mildew", "powdery", "white powder", "white powdery", "powdery white"
    ],
    "Early Blight": [
        "early blight", "alternaria", "concentric rings", "target board", "target spot"
    ],
    "Late Blight": [
        "late blight", "phytophthora", "water-soaked", "water soaked", "pale halo", "pale margins"
    ]
}

# Negation indicators common in expert descriptions for healthy crops
NEGATION_PATTERNS = [
    r"\bno\s+(?:evidence\s+of\s+)?(?:blight|mildew|spots?|lesions?|disease|symptoms?)\b",
    r"\bfree\s+(?:of|from)\s+(?:blight|mildew|spots?|lesions?|disease|symptoms?)\b",
    r"\bwithout\s+(?:any\s+)?(?:blight|mildew|spots?|lesions?|disease|symptoms?)\b",
    r"\babsent\b",
    r"\bnot\s+observed\b",
    r"\bno\s+visible\s+(?:blight|mildew|spots?|lesions?)\b"
]


def sanitize_text(text: str) -> str:
    """
    Applies the sanitization function S(d_i) to remove direct disease-label expressions.
    
    Args:
        text (str): Raw input description.
        
    Returns:
        str: Sanitized text description.
    """
    if not isinstance(text, str):
        return ""
    
    sanitized = text
    for pattern, replacement in DISEASE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Clean up double spaces or brackets if needed
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def audit_text_leakage(texts: List[str], labels: List[str]) -> Dict[str, Dict]:
    """
    Performs a diagnostic leakage audit across classes.
    Checks whether texts contain explicit disease terms corresponding to their class.
    
    Args:
        texts: List of textual descriptions (raw or sanitized).
        labels: List of class labels ('Healthy', 'Powdery Mildew', 'Early Blight', 'Late Blight').
        
    Returns:
        Dictionary with per-class leakage counts, total samples, and percentage.
    """
    class_names = ["Healthy", "Powdery Mildew", "Early Blight", "Late Blight"]
    audit_results = {}
    
    for cls in class_names:
        cls_texts = [text for text, label in zip(texts, labels) if label == cls]
        total_cls = len(cls_texts)
        if total_cls == 0:
            continue
            
        leaked_count = 0
        negation_count = 0
        keywords = AUDIT_KEYWORDS.get(cls, [])
        
        for text in cls_texts:
            text_lower = text.lower()
            # Check for keyword occurrence
            has_term = any(kw in text_lower for kw in keywords)
            
            # Check if this occurrence is part of a negation (critical for Healthy class!)
            is_negated = any(re.search(neg, text_lower) for neg in NEGATION_PATTERNS)
            if is_negated:
                negation_count += 1
                
            if has_term:
                leaked_count += 1
                
        audit_results[cls] = {
            "total_records": total_cls,
            "leaked_records": leaked_count,
            "leakage_percentage": (leaked_count / total_cls) * 100.0,
            "negated_records": negation_count,
            "negation_percentage": (negation_count / total_cls) * 100.0 if total_cls > 0 else 0.0
        }
        
    return audit_results


def compare_raw_vs_sanitized_leakage(raw_texts: List[str], sanitized_texts: List[str], labels: List[str]):
    """
    Compares leakage rates before and after text sanitization.
    """
    raw_audit = audit_text_leakage(raw_texts, labels)
    san_audit = audit_text_leakage(sanitized_texts, labels)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC LEAKAGE AUDIT COMPARISON (RAW vs. SANITIZED)")
    print("="*70)
    print(f"{'Class':<16} | {'Total':<6} | {'Raw Leaked (%)':<16} | {'Sanitized Leaked (%)':<20}")
    print("-" * 70)
    for cls in raw_audit:
        r_info = raw_audit[cls]
        s_info = san_audit.get(cls, {})
        print(f"{cls:<16} | {r_info['total_records']:<6} | "
              f"{r_info['leaked_records']}/{r_info['total_records']} ({r_info['leakage_percentage']:.2f}%)   | "
              f"{s_info.get('leaked_records', 0)}/{s_info.get('total_records', 1)} ({s_info.get('leakage_percentage', 0.0):.2f}%)")
    print("="*70)
    return raw_audit, san_audit
