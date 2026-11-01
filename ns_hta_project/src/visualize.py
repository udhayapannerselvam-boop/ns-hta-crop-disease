"""
visualize.py - Publication-quality plotting and visualization routines.

Generates:
1. Fig 1: Confusion Matrix (raw counts and row-normalized percentages with legible fonts).
2. Fig 2: Class-wise F1-Scores Bar Chart (with exact numeric labels on top of bars).
3. Fig 3: Representation-Level Attribution Bar Chart (Eq. 14).
4. Fig 4: End-to-End System Architecture Diagram.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


# Set clean publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 15
})


def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
    prefix: str = "ns_hta_pairwise"
):
    """
    Plots both raw count confusion matrix and row-normalized percentage confusion matrix.
    Uses large, legible fonts as requested in the review.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Raw Count Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 13, "weight": "bold"}, ax=ax
    )
    ax.set_title("NS-HTA Pairwise Confusion Matrix (Pooled OOF)", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual Class", fontsize=12, labelpad=8)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    raw_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_counts.png")
    plt.savefig(raw_path, dpi=300)
    plt.close()
    
    # 2. Row-Normalized (Row-%) Heatmap
    cm_row_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12) * 100.0
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # Custom annotations combining count and percentage
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_row_pct[i, j]:.1f}%)"
            
    sns.heatmap(
        cm_row_pct, annot=annot_matrix, fmt="", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 11, "weight": "bold"}, ax=ax, vmin=0, vmax=100
    )
    ax.set_title("NS-HTA Normalized Confusion Matrix (Row %)", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual Class (Recall)", fontsize=12, labelpad=8)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    norm_path = os.path.join(output_dir, f"{prefix}_confusion_matrix_normalized.png")
    plt.savefig(norm_path, dpi=300)
    plt.close()
    
    print(f"[Visualization] Saved confusion matrices to {raw_path} and {norm_path}")


def plot_class_f1_scores(
    per_class_dict: Dict[str, Dict],
    output_dir: str,
    filename: str = "fig2_class_f1.png"
):
    """
    Plots bar chart of class-wise F1 scores with exact numeric values on top of bars.
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = list(per_class_dict.keys())
    f1_scores = [per_class_dict[c]["f1"] for c in classes]
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax.bar(classes, f1_scores, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    
    # Add exact numeric values above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )
        
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1-Score (%)", fontsize=13, labelpad=8)
    ax.set_title("Class-wise F1-Scores of Sanitized NS-HTA Pairwise Model", fontsize=14, pad=12, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved Class F1 bar chart to {save_path}")


def plot_attribution_weights(
    attribution_dict: Dict[str, float],
    output_dir: str,
    filename: str = "fig3_attribution_weights.png"
):
    """
    Plots bar chart of representation-level attribution (Eq. 14).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for the main representation blocks
    labels = ["Text Block (h_T)", "Symbolic Block (h_S)", "Interaction Block (h_TS)"]
    raw_vals = [
        attribution_dict.get("A_text (Raw Mean |W|)", 0.0),
        attribution_dict.get("A_symbolic (Raw Mean |W|)", 0.0),
        attribution_dict.get("A_interaction (Raw Mean |W|)", 0.0)
    ]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, raw_vals, color=["#4A90E2", "#50E3C2", "#F5A623"], width=0.5, edgecolor="black", linewidth=1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )
        
    ax.set_ylabel("Mean Absolute Weight Magnitude (A_k)", fontsize=12)
    ax.set_title("Representation-Level Classifier Attribution (Eq. 14)", fontsize=13, weight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved attribution weights to {save_path}")


def generate_architecture_diagram(output_dir: str, filename: str = "fig4_architecture.png"):
    """
    Generates a high-quality publication schematic of the NS-HTA architecture:
    Text -> Sanitization -> Embedding (384) + 5 Rules -> 32-d Projections -> Hadamard -> 96-d -> Softmax.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc="#E8F0FE", ec="#1A73E8", lw=1.5)
    box_sym = dict(boxstyle="round,pad=0.5", fc="#E6F4EA", ec="#137333", lw=1.5)
    box_fusion = dict(boxstyle="round,pad=0.5", fc="#FEF7E0", ec="#EA8600", lw=1.5)
    box_head = dict(boxstyle="round,pad=0.5", fc="#FCE8E6", ec="#C5221F", lw=1.5)
    
    # 1. Text Pipeline
    ax.text(0.12, 0.78, "Raw Symptom Text\n(d_i)", ha="center", va="center", bbox=bbox_props, fontsize=10, weight="bold")
    ax.annotate("", xy=(0.26, 0.78), xytext=(0.19, 0.78), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.text(0.34, 0.78, "Text Sanitization\nS(d_i)", ha="center", va="center", bbox=bbox_props, fontsize=10)
    ax.annotate("", xy=(0.46, 0.78), xytext=(0.41, 0.78), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.text(0.55, 0.78, "Frozen Encoder E(·)\n(all-MiniLM-L6-v2)\n[x_T in R^384]", ha="center", va="center", bbox=bbox_props, fontsize=9)
    ax.annotate("", xy=(0.68, 0.78), xytext=(0.63, 0.78), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.text(0.77, 0.78, "Text Latent Projection\nh_T = ReLU(W_T x_T + b_T)\n[h_T in R^32]", ha="center", va="center", bbox=bbox_props, fontsize=9)
    
    # 2. Symbolic Pipeline
    ax.annotate("", xy=(0.34, 0.32), xytext=(0.34, 0.70), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--"))
    ax.text(0.34, 0.25, "5 Predefined\nSymptom Rules", ha="center", va="center", bbox=box_sym, fontsize=10)
    ax.annotate("", xy=(0.46, 0.25), xytext=(0.41, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.text(0.55, 0.25, "Symbolic Vector\n[x_S in {0,1}^5]", ha="center", va="center", bbox=box_sym, fontsize=9)
    ax.annotate("", xy=(0.68, 0.25), xytext=(0.63, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    ax.text(0.77, 0.25, "Symbolic Projection\nh_S = ReLU(W_S x_S + b_S)\n[h_S in R^32]", ha="center", va="center", bbox=box_sym, fontsize=9)
    
    # 3. Fusion & Interaction
    ax.annotate("", xy=(0.88, 0.60), xytext=(0.77, 0.70), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.88, 0.44), xytext=(0.77, 0.33), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    fusion_text = (
        "Pairwise Hyper-Tensor Alignment\n"
        "------------------------------------\n"
        "• Interaction: h_TS = h_T ⊙ h_S  (R^32)\n"
        "• Fused: z_TS = [h_T; h_S; h_TS]  (R^96)"
    )
    ax.text(0.88, 0.52, fusion_text, ha="center", va="center", bbox=box_fusion, fontsize=9.5, weight="bold")
    
    # 4. Classification Head
    ax.annotate("", xy=(0.88, 0.30), xytext=(0.88, 0.40), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.88, 0.18, "Classification Layer (4-Class Softmax)\nHealthy | Powdery Mildew | Early Blight | Late Blight\nParameters: 12,900 trainable weights",
            ha="center", va="center", bbox=box_head, fontsize=9.5, weight="bold")
            
    ax.set_title("NS-HTA Architecture: Zero-Image Text & Symbolic Pairwise Alignment", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved architecture schematic to {save_path}")
