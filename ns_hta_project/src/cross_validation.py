"""
cross_validation.py - 5-Fold Stratified Group Cross-Validation and Metric Computation.

Implements:
1. Stratified Group K-Fold splitting by image_id with zero-overlap audit.
2. Equations (16)-(19):
   - Accuracy (16)
   - Per-class precision, recall, and F1 (17)
   - Macro-F1 (18)
   - Weighted-F1 (19)
3. Pooled out-of-fold metrics and fold-level (Mean +/- SD) metrics.
4. Confusion matrix computation (raw counts + normalized row-%).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix
)


def get_stratified_group_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates 5-fold splits stratified by class label and grouped by image_id.
    Performs zero-overlap validation audit across all folds.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    splits = []
    
    print("\n" + "="*70)
    print("5-FOLD STRATIFIED GROUP CROSS-VALIDATION AUDIT")
    print("="*70)
    
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, df["class_idx"], df["image_id"])):
        train_groups = set(df.iloc[train_idx]["image_id"])
        test_groups = set(df.iloc[test_idx]["image_id"])
        overlap = train_groups.intersection(test_groups)
        
        assert len(overlap) == 0, f"Critical leakage detected in fold {fold_idx}: {overlap}"
        
        test_counts = df.iloc[test_idx]["class_idx"].value_counts().to_dict()
        print(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Test={len(test_idx)} records | "
              f"Image groups: Train={len(train_groups)}, Test={len(test_groups)} (Overlap: {len(overlap)}) | "
              f"Test classes: {test_counts}")
        splits.append((train_idx, test_idx))
        
    print("="*70)
    return splits


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Computes Accuracy, Weighted F1, Macro F1, and per-class metrics.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, average="weighted") * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12) * 100.0
    
    per_class_metrics = {}
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(support))]
        
    for i, name in enumerate(class_names):
        per_class_metrics[name] = {
            "precision": float(p_per_class[i] * 100.0),
            "recall": float(r_per_class[i] * 100.0),
            "f1": float(f1_per_class[i] * 100.0),
            "support": int(support[i])
        }
        
    return {
        "accuracy": float(acc),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm
    }


def run_cross_validation_experiment(
    model_name: str,
    train_eval_fn,
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Executes cross-validation across all 5 folds for a given model.
    Collects pooled out-of-fold predictions and fold-level statistics.
    """
    N = len(df)
    oof_predictions = np.zeros(N, dtype=int)
    oof_probabilities = np.zeros((N, len(class_names)), dtype=float)
    fold_macro_f1s = []
    fold_accuracies = []
    fold_weighted_f1s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Call training and evaluation function for the fold
        preds, probs = train_eval_fn(train_idx, test_idx, fold_idx)
        
        oof_predictions[test_idx] = preds
        if probs is not None:
            oof_probabilities[test_idx] = probs
            
        fold_y_true = df.iloc[test_idx]["class_idx"].values
        fold_metrics = evaluate_predictions(fold_y_true, preds, class_names)
        
        fold_macro_f1s.append(fold_metrics["macro_f1"])
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_weighted_f1s.append(fold_metrics["weighted_f1"])
        
    y_true_all = df["class_idx"].values
    pooled_metrics = evaluate_predictions(y_true_all, oof_predictions, class_names)
    
    # 95% Bootstrap Confidence Intervals over out-of-fold samples (1,000 resamples)
    rng = np.random.RandomState(42)
    boot_macro_f1s = []
    n_samples = len(y_true_all)
    for _ in range(1000):
        boot_idx = rng.choice(n_samples, n_samples, replace=True)
        b_true = y_true_all[boot_idx]
        b_pred = oof_predictions[boot_idx]
        boot_macro_f1s.append(f1_score(b_true, b_pred, average="macro", zero_division=0) * 100.0)
    ci_95 = (float(np.percentile(boot_macro_f1s, 2.5)), float(np.percentile(boot_macro_f1s, 97.5)))
    
    results = {
        "model_name": model_name,
        "pooled_accuracy": pooled_metrics["accuracy"],
        "pooled_weighted_f1": pooled_metrics["weighted_f1"],
        "pooled_macro_f1": pooled_metrics["macro_f1"],
        "pooled_macro_f1_ci95": ci_95,
        "fold_macro_f1_mean": float(np.mean(fold_macro_f1s)),
        "fold_macro_f1_sd": float(np.std(fold_macro_f1s)),
        "fold_acc_mean": float(np.mean(fold_accuracies)),
        "fold_acc_sd": float(np.std(fold_accuracies)),
        "per_class": pooled_metrics["per_class"],
        "confusion_matrix": pooled_metrics["confusion_matrix"],
        "confusion_matrix_norm": pooled_metrics["confusion_matrix_norm"],
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities
    }
    return results
