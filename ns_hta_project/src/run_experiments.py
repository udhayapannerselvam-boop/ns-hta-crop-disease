"""
run_experiments.py - Master evaluation pipeline reproducing Table II, Figures, and Audits.

Executes:
1. Benchmark dataset loading and text sanitization.
2. Diagnostic leakage audit (raw vs. sanitized).
3. Symbolic 5-rule extraction and 384-d text embedding encoding.
4. 5-Fold Stratified Group Cross-Validation (by image_id).
5. Evaluation of all 9 paper configurations + enhanced reviewer baselines.
6. Representation attribution and computational profiling.
7. Publication figure generation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_dataset, CLASSES
from sanitization import sanitize_text, compare_raw_vs_sanitized_leakage
from symbolic_rules import batch_extract_symbolic_features, get_rule_descriptions_table
from embeddings import TextEmbeddingGenerator
from models import NSHTAPairwiseModel
from baselines import get_baseline_models
from cross_validation import get_stratified_group_splits, run_cross_validation_experiment
from profiler import profile_model_resources, print_resource_profile
from visualize import (
    plot_confusion_matrices,
    plot_class_f1_scores,
    plot_attribution_weights,
    generate_architecture_diagram
)


def run_full_pipeline(output_dir: str = None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    figures_dir = os.path.join(project_root, "figures")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("      NS-HTA EMPIRICAL BENCHMARK & REPRODUCIBILITY SUITE")
    print("="*80)
    
    # 1. Load Dataset
    csv_path = os.path.join(data_dir, "crop_disease_370.csv")
    df = load_dataset(csv_path)
    print(f"\n[1/7] Dataset Loaded: {len(df)} records across {len(CLASSES)} classes.")
    print(df["label"].value_counts().to_string())
    
    # 2. Text Sanitization
    print("\n[2/7] Running Lexical Text Sanitization...")
    df["sanitized_description"] = df["description"].apply(sanitize_text)
    
    # Print sanitization examples
    print("\nSanitization Example (Original -> Sanitized):")
    sample_row = df[df["label"] == "Powdery Mildew"].iloc[0]
    print(f"  Raw      : {sample_row['description']}")
    print(f"  Sanitized: {sample_row['sanitized_description']}")
    
    # 3. Diagnostic Leakage Audit
    print("\n[3/7] Performing Diagnostic Leakage Audit...")
    raw_texts = df["description"].tolist()
    san_texts = df["sanitized_description"].tolist()
    labels = df["label"].tolist()
    raw_audit, san_audit = compare_raw_vs_sanitized_leakage(raw_texts, san_texts, labels)
    
    # 4. Feature Extraction
    print("\n[4/7] Extracting Symbolic & Textual Features...")
    emb_gen = TextEmbeddingGenerator(model_name="all-MiniLM-L6-v2", cache_dir=data_dir)
    
    # Text embeddings: 384 dimensions
    X_T_raw = emb_gen.encode(raw_texts, use_cache=True, cache_prefix="raw_text")
    X_T_san = emb_gen.encode(san_texts, use_cache=True, cache_prefix="san_text")
    
    # Symbolic features: 5 dimensions (extracted on sanitized text)
    X_S = batch_extract_symbolic_features(san_texts)
    
    # Combined representations: 389 dimensions
    X_ZI_raw = np.concatenate([X_T_raw, X_S], axis=1)
    X_ZI_san = np.concatenate([X_T_san, X_S], axis=1)
    
    print(f"  Text features (raw) shape     : {X_T_raw.shape}")
    print(f"  Text features (sanitized) shape: {X_T_san.shape}")
    print(f"  Symbolic features shape       : {X_S.shape}")
    print(f"  Combined input shape (X_ZI)   : {X_ZI_san.shape}")
    
    # 5. Stratified Group Cross-Validation
    print("\n[5/7] Preparing 5-Fold Stratified Group Cross-Validation...")
    splits = get_stratified_group_splits(df, n_splits=5, random_seed=42)
    
    # 6. Model Training & Benchmarking
    print("\n[6/7] Running Cross-Validation for All Evaluated Models...")
    all_results = []
    
    # 6.1 Baseline Models
    baselines = get_baseline_models(random_seed=42)
    
    # Map each baseline to its appropriate input features
    model_configs = [
        ("Text only (raw)", "raw_text_only", baselines["Text only (raw)"]),
        ("Text + symbolic (Logistic Regression)", "san_combined", baselines["Text + symbolic (Logistic Regression)"]),
        ("Text only (sanitized)", "san_text_only", baselines["Text only (sanitized)"]),
        ("Text + symbolic (Linear SVM)", "san_combined", baselines["Text + symbolic (Linear SVM)"]),
        ("Symbolic only", "symbolic_only", baselines["Symbolic only"]),
        ("Text + symbolic (Random Forest)", "san_combined", baselines["Text + symbolic (Random Forest)"]),
        ("Majority baseline", "symbolic_only", baselines["Majority baseline"]),
        # Reviewer additions
        ("Text + symbolic (LR Balanced)", "san_combined", baselines["Text + symbolic (LR Balanced)"]),
        ("Text + symbolic (Plain MLP 389-d)", "san_combined", baselines["Text + symbolic (Plain MLP 389-d)"]),
        ("TF-IDF + Logistic Regression", "tfidf_san", baselines["TF-IDF + Logistic Regression"]),
    ]
    
    for name, input_type, clf in model_configs:
        def make_train_fn(inp_t, c):
            def train_eval(train_idx, test_idx, fold_idx):
                if inp_t == "raw_text_only":
                    X_tr, X_te = X_T_raw[train_idx], X_T_raw[test_idx]
                elif inp_t == "san_text_only":
                    X_tr, X_te = X_T_san[train_idx], X_T_san[test_idx]
                elif inp_t == "symbolic_only":
                    X_tr, X_te = X_S[train_idx], X_S[test_idx]
                elif inp_t == "san_combined":
                    X_tr, X_te = X_ZI_san[train_idx], X_ZI_san[test_idx]
                elif inp_t == "tfidf_san":
                    X_tr = [san_texts[i] for i in train_idx]
                    X_te = [san_texts[i] for i in test_idx]
                    
                y_tr = df.iloc[train_idx]["class_idx"].values
                c.fit(X_tr, y_tr)
                preds = c.predict(X_te)
                probs = c.predict_proba(X_te) if hasattr(c, "predict_proba") else None
                return preds, probs
            return train_eval
            
        res = run_cross_validation_experiment(
            name, make_train_fn(input_type, clf), df, splits, CLASSES
        )
        all_results.append(res)
        
    # 6.2 NS-HTA Pairwise Model (Sanitized)
    def train_eval_ns_hta(train_idx, test_idx, fold_idx):
        model = NSHTAPairwiseModel(
            d_text=384, d_sym=5, d_latent=32, num_classes=4,
            with_interaction=True, learning_rate=0.015, weight_decay=1e-4,
            random_seed=42 + fold_idx, use_class_weights=False
        )
        model.fit(
            X_T_san[train_idx], X_S[train_idx],
            df.iloc[train_idx]["class_idx"].values,
            epochs=50, batch_size=16
        )
        preds = model.predict(X_T_san[test_idx], X_S[test_idx])
        probs = model.predict_proba(X_T_san[test_idx], X_S[test_idx])
        return preds, probs
        
    ns_hta_res = run_cross_validation_experiment(
        "NS-HTA pairwise (sanitized)", train_eval_ns_hta, df, splits, CLASSES
    )
    all_results.append(ns_hta_res)
    
    # 6.3 Pairwise Model Without Interaction (Ablation)
    def train_eval_ns_no_int(train_idx, test_idx, fold_idx):
        model = NSHTAPairwiseModel(
            d_text=384, d_sym=5, d_latent=32, num_classes=4,
            with_interaction=False, learning_rate=0.015, weight_decay=1e-4,
            random_seed=42 + fold_idx, use_class_weights=False
        )
        model.fit(
            X_T_san[train_idx], X_S[train_idx],
            df.iloc[train_idx]["class_idx"].values,
            epochs=50, batch_size=16
        )
        preds = model.predict(X_T_san[test_idx], X_S[test_idx])
        probs = model.predict_proba(X_T_san[test_idx], X_S[test_idx])
        return preds, probs
        
    ns_no_int_res = run_cross_validation_experiment(
        "Pairwise without interaction", train_eval_ns_no_int, df, splits, CLASSES
    )
    all_results.append(ns_no_int_res)
    
    # 6.4 NS-HTA Pairwise Model with Class Weighting (Reviewer Enhancement)
    def train_eval_ns_hta_balanced(train_idx, test_idx, fold_idx):
        model = NSHTAPairwiseModel(
            d_text=384, d_sym=5, d_latent=32, num_classes=4,
            with_interaction=True, learning_rate=0.01, weight_decay=1e-4,
            random_seed=42 + fold_idx, use_class_weights=True
        )
        model.fit(
            X_T_san[train_idx], X_S[train_idx],
            df.iloc[train_idx]["class_idx"].values,
            epochs=60, batch_size=16
        )
        preds = model.predict(X_T_san[test_idx], X_S[test_idx])
        probs = model.predict_proba(X_T_san[test_idx], X_S[test_idx])
        return preds, probs
        
    ns_hta_bal_res = run_cross_validation_experiment(
        "NS-HTA pairwise (Balanced Loss)", train_eval_ns_hta_balanced, df, splits, CLASSES
    )
    all_results.append(ns_hta_bal_res)
    
    # 7. Print Comparative Table II
    print("\n" + "="*115)
    print("TABLE II: COMPARATIVE PERFORMANCE OF EVALUATED MODELS (WITH 95% BOOTSTRAP CI)")
    print("="*115)
    print(f"{'Model':<38} | {'Accuracy (%)':<12} | {'Weighted F1':<12} | {'Macro F1 (%)':<12} | {'95% CI':<16} | {'Fold Macro F1 (Mean ± SD)':<22}")
    print("-" * 115)
    
    # Sort order matching Table II in paper
    table_order = [
        "Text only (raw)",
        "Text + symbolic (Logistic Regression)",
        "Text only (sanitized)",
        "Text + symbolic (Linear SVM)",
        "Symbolic only",
        "NS-HTA pairwise (sanitized)",
        "Pairwise without interaction",
        "Text + symbolic (Random Forest)",
        "Majority baseline",
        # Enhancements
        "Text + symbolic (LR Balanced)",
        "Text + symbolic (Plain MLP 389-d)",
        "TF-IDF + Logistic Regression",
        "NS-HTA pairwise (Balanced Loss)"
    ]
    
    res_dict = {r["model_name"]: r for r in all_results}
    
    for m_name in table_order:
        if m_name in res_dict:
            r = res_dict[m_name]
            ci_str = f"[{r['pooled_macro_f1_ci95'][0]:.1f}, {r['pooled_macro_f1_ci95'][1]:.1f}]"
            print(f"{r['model_name']:<38} | {r['pooled_accuracy']:<12.2f} | {r['pooled_weighted_f1']:<12.2f} | "
                  f"{r['pooled_macro_f1']:<12.2f} | {ci_str:<16} | {r['fold_macro_f1_mean']:.2f} ± {r['fold_macro_f1_sd']:.2f}")
    print("="*115)
    
    # Per-Class Analysis for Key Models
    print("\n" + "="*80)
    print("PER-CLASS METRIC COMPARISON (POOLED OUT-OF-FOLD)")
    print("="*80)
    for target_m in ["NS-HTA pairwise (sanitized)", "Text + symbolic (Logistic Regression)", "NS-HTA pairwise (Balanced Loss)"]:
        if target_m in res_dict:
            r = res_dict[target_m]
            print(f"\nModel: {target_m}")
            print(f"{'Class':<16} | {'Precision (%)':<14} | {'Recall (%)':<12} | {'F1-Score (%)':<14} | {'Support':<8}")
            print("-" * 70)
            for c_name, c_data in r["per_class"].items():
                print(f"{c_name:<16} | {c_data['precision']:<14.2f} | {c_data['recall']:<12.2f} | {c_data['f1']:<14.2f} | {c_data['support']:<8}")
                
    # 8. Representation Attribution Calculation (Eq. 14 + Causal Ablation)
    print("\n[7/7] Computing Representation-Level Attribution (Eq. 14 & Causal Occlusion)...")
    final_ns_model = NSHTAPairwiseModel(d_text=384, d_sym=5, d_latent=32, num_classes=4, with_interaction=True)
    final_ns_model.fit(X_T_san, X_S, df["class_idx"].values, epochs=50)
    attribution = final_ns_model.compute_representation_attribution(X_T_san, X_S, df["class_idx"].values)
    
    print("\nRepresentation Attribution Scores:")
    for k, v in attribution.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                print(f"    - {sub_k:<35}: {sub_v:.2f}")
        else:
            print(f"  {k:<35}: {v:.4f}")
        
    # 9. Computational Resource Profiling
    profile = profile_model_resources(final_ns_model, X_T_san, X_S)
    print_resource_profile(profile)
    
    # 10. Generate Figures
    print("\nGenerating publication figures...")
    ns_hta_eval = res_dict["NS-HTA pairwise (sanitized)"]
    plot_confusion_matrices(ns_hta_eval["confusion_matrix"], CLASSES, figures_dir, prefix="fig1_ns_hta")
    plot_class_f1_scores(ns_hta_eval["per_class"], figures_dir, filename="fig2_class_f1.png")
    plot_attribution_weights(attribution, figures_dir, filename="fig3_attribution_weights.png")
    generate_architecture_diagram(figures_dir, filename="fig4_architecture.png")
    
    # Save experiment report to JSON
    report_path = os.path.join(figures_dir, "benchmark_summary.json")
    summary_to_save = {
        "raw_leakage_audit": raw_audit,
        "sanitized_leakage_audit": san_audit,
        "models": {
            r["model_name"]: {
                "accuracy": r["pooled_accuracy"],
                "weighted_f1": r["pooled_weighted_f1"],
                "macro_f1": r["pooled_macro_f1"],
                "fold_macro_f1_mean": r["fold_macro_f1_mean"],
                "fold_macro_f1_sd": r["fold_macro_f1_sd"],
                "per_class": r["per_class"]
            }
            for r in all_results
        },
        "attribution": attribution,
        "resource_profile": profile
    }
    with open(report_path, "w") as f:
        json.dump(summary_to_save, f, indent=2)
    print(f"\nBenchmark summary report saved to {report_path}")
    print("All tasks completed successfully!")


if __name__ == "__main__":
    run_full_pipeline()
