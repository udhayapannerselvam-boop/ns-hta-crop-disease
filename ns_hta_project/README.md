# NS-HTA: Neuro-Symbolic Hyper-Tensor Alignment for Zero-Image Crop Disease Classification

A reproducible, resource-bounded empirical evaluation suite for zero-image crop disease classification from textual descriptions and symbolic symptom features.

## Project Structure
```
ns_hta_project/
├── data/
│   ├── crop_disease_370.csv      # 370-record benchmark dataset (4 classes)
│   ├── raw_text_*.npz            # Cached 384-d raw text embeddings
│   └── san_text_*.npz            # Cached 384-d sanitized text embeddings
├── src/
│   ├── dataset.py                # Dataset loader & stratified group generation
│   ├── sanitization.py           # Lexical sanitization & diagnostic leakage audit
│   ├── symbolic_rules.py         # 5 rule-derived binary symptom extractors
│   ├── embeddings.py             # 384-dimensional frozen sentence embeddings
│   ├── models.py                 # NS-HTA PyTorch & NumPy neural architectures
│   ├── baselines.py              # Scikit-learn baselines (LR, SVM, RF, MLP, TF-IDF)
│   ├── cross_validation.py       # 5-fold stratified group CV & metric reporting
│   ├── profiler.py               # Parameter count, latency, and RAM profiler
│   ├── visualize.py              # Publication-grade plotting routines
│   └── run_experiments.py        # Master benchmarking script
├── figures/
│   ├── fig1_ns_hta_confusion_matrix_counts.png
│   ├── fig1_ns_hta_confusion_matrix_normalized.png
│   ├── fig2_class_f1.png
│   ├── fig3_attribution_weights.png
│   ├── fig4_architecture.png
│   └── benchmark_summary.json
├── paper/
│   ├── paper_revision.tex        # Complete camera-ready IEEE conference LaTeX source
│   └── PAPER_WRITING_GUIDE.md    # Section-by-section writing and revision guide
├── requirements.txt              # Python dependencies
├── run_all.py                    # One-command execution script
└── README.md
```

## Quick Start

### 1. Run the Full Benchmark
To execute the end-to-end benchmark, run all 12 models across 5 folds, audit data leakage, profile hardware resources, and generate all publication figures:
```powershell
python run_all.py
```

### 2. Evaluated Classes (370 samples)
- **Healthy**: 303 (81.89%)
- **Powdery Mildew**: 43 (11.62%)
- **Early Blight**: 14 (3.78%)
- **Late Blight**: 10 (2.70%)

### 3. Key Findings
1. **Diagnostic Text Leakage**: Up to 85.71% of expert descriptions contain explicit disease labels or pathogens, artificially inflating raw-text classifier performance. 34.32% of healthy descriptions contain negation phrases (e.g., "no blight or mildew observed").
2. **Resource-Bounded Efficiency**: NS-HTA requires only **12,900 trainable parameters**, consumes **50.39 KB** of weight memory, and executes inference in **0.036 ms** per sample on a single CPU core (27,398 samples/sec).
3. **Imbalance Optimization**: Unweighted pairwise models suffer from minority gradient starvation (Macro F1 = 56.53%). Introducing inverse-frequency class weighting boosts Macro F1 to **98.64%**, matching balanced linear baselines.
