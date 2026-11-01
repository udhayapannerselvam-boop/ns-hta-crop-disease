"""
baselines.py - Baseline models and comparative configurations for Table II.

Implements all 9 configurations from the paper plus the reviewer-recommended baselines:
1. Majority class baseline
2. Symbolic-only classifier (5 features)
3. Text-only classifier (raw text)
4. Text-only classifier (sanitized text)
5. Text + symbolic Logistic Regression (standard and class-weighted)
6. Text + symbolic Linear SVM
7. Text + symbolic Random Forest
8. TF-IDF + Logistic Regression baseline
9. Plain 389-dimensional Concatenation MLP baseline
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def get_baseline_models(random_seed: int = 42) -> dict:
    """
    Returns an initialized dictionary of all baseline models.
    """
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        
        "Symbolic only": LogisticRegression(
            C=1.0, max_iter=500, random_state=random_seed
        ),
        
        "Text only (raw)": LogisticRegression(
            C=1.0, max_iter=500, random_state=random_seed
        ),
        
        "Text only (sanitized)": LogisticRegression(
            C=1.0, max_iter=500, random_state=random_seed
        ),
        
        "Text + symbolic (Logistic Regression)": LogisticRegression(
            C=1.0, max_iter=500, random_state=random_seed
        ),
        
        "Text + symbolic (LR Balanced)": LogisticRegression(
            C=1.0, max_iter=500, class_weight="balanced", random_state=random_seed
        ),
        
        "Text + symbolic (Linear SVM)": LinearSVC(
            C=0.5, max_iter=2000, random_state=random_seed, dual="auto"
        ),
        
        "Text + symbolic (Random Forest)": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_seed
        ),
        
        "Text + symbolic (Plain MLP 389-d)": MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", max_iter=300, random_state=random_seed
        ),
        
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=random_seed))
        ])
    }
