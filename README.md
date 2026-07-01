# Financial Sentiment Analysis — Advanced NLP Pipeline

> A comprehensive project comparing classical machine learning approaches with state-of-the-art transformer-based models for classifying sentiment in financial text.

---

## 📋 Project Overview

This project evaluates and contrasts multiple families of approaches for financial sentiment classification, featuring a massive **12-step automated pipeline**.

| Approach | Models | Key Characteristics |
|---|---|---|
| **Traditional ML** | LogisticRegression, LinearSVC, Naive Bayes + SMOTE | Fast, interpretable, lightweight |
| **Core Transformers** | FinBERT, BERT-base, FinBERT (INT8) | Context-aware, domain-adapted |
| **Advanced HF Models** | DistilBERT, RoBERTa, DeBERTa-v3, Twitter-RoBERTa | SOTA architectures, broad comparisons |
| **PEFT & Zero-Shot** | FinBERT-LoRA (4-bit), BART-large-MNLI | Parameter efficient, zero-shot baselines |
| **Ensemble Methods** | Weighted Voting Ensembles | High accuracy, variance reduction |

**Dataset:** `data.csv` — 5,844 sentences labeled `positive`, `negative`, or `neutral`.

**Split:** 70% train / 15% validation / 15% test (stratified).

---

## 🎯 Research Objectives

1. Build a reproducible ML pipeline with TF-IDF, SMOTE, and classical classifiers.
2. Fine-tune multiple transformer models (FinBERT, DeBERTa, etc.) adapted to financial text.
3. Utilize Parameter-Efficient Fine-Tuning (LoRA) and 4-bit Quantization for resource-constrained environments.
4. Evaluate Zero-Shot capabilities using NLI models (BART-large).
5. Compare approaches across macro-F1, training time, inference latency, GPU memory usage, and out-of-distribution (OOD) performance.
6. Conduct deep interpretability and error analysis (EDA, UMAP, TF-IDF features).

---

## 🗂️ Project Structure

```
project/
├── data/                  # Processed train/val/test splits
├── models/                # Saved model artifacts & tokenizers
├── results/               # Evaluation outputs, plots, and JSONs
├── src/
│   ├── config.py           # Paths & API Keys
│   ├── preprocessing.py    # Text cleaning, NLTK stopword filtering
│   ├── eda_analysis.py     # Class distribution, UMAP, WordClouds, TF-IDF
│   ├── ml_pipeline.py      # TF-IDF + LogReg/SVM + SMOTE
│   ├── transformer_pipeline.py  # FinBERT / BERT / Quantized INT8
│   ├── additional_models.py# DistilBERT, RoBERTa, DeBERTa, Twitter
│   ├── lora_pipeline.py    # PEFT LoRA + BitsAndBytes 4-bit
│   ├── zero_shot.py        # BART-large-MNLI Zero-Shot inference
│   ├── ensemble.py         # Weighted Voting
│   ├── error_analysis.py   # Hard sample extraction, disagreement matrix
│   ├── ood_evaluation.py   # NewsAPI real-time OOD validation
│   ├── robustness.py       # Multi-level Synonym + Typo noise injection
│   ├── evaluation.py       # Metrics, radar charts, cost-quality tradeoff
│   └── main.py             # 12-Step Orchestrator
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Hardware Note:** An NVIDIA GPU (e.g., RTX 4070) is strongly recommended for 4-bit Quantization, LoRA, and Transformer pipelines.

### 2 — Configure API Keys

In `src/config.py`, add your free NewsAPI key for OOD Evaluation:
```python
NEWSAPI_KEY = "your_newsapi_key_here"
```

### 3 — Run the Full 12-Step Pipeline

```bash
# Make sure your terminal is in the project root
python src/main.py
```

The orchestrator will automatically execute:
1. Preprocessing
2. EDA & Interpretability
3. Traditional ML & SMOTE
4. Core Transformers
5. Additional HF Models
6. LoRA Fine-Tuning
7. Zero-Shot Baseline
8. Ensemble Methods
9. Error Analysis
10. OOD Evaluation
11. Robustness Evaluation
12. Comprehensive Evaluation & Visualization

---

## 📊 Evaluation & Visualizations

The pipeline generates advanced visualizations in the `results/` folder:
- **Radar Charts**: Multi-dimensional comparison of top models.
- **Cost-Quality Tradeoff**: Scatter plot of F1 Score vs Inference Latency (bubble size = GPU Memory).
- **UMAP Projections**: 2D embeddings of TF-IDF vectors.
- **Degradation Curves**: Model performance decay under varying noise levels.
- **Error Analysis Heatmaps**: Model disagreement matrices and hard sample extractions.
- **OOD Shift Plots**: In-distribution vs Real-time NewsAPI prediction shifts.

---

## 📝 Key Design Decisions

- **Handling Imbalance:** Applied SMOTE for ML models and Class-Weighted CrossEntropyLoss for Transformers.
- **Robustness:** Evaluated up to 20% noise injection (WordNet synonym replacements + character-level typos).
- **Efficiency Metrics:** Tracked exact training seconds, per-sample inference latency (ms), and peak GPU memory footprint (MB).
- **Parameter Efficiency:** Integrated `peft` (LoRA) and `bitsandbytes` (NF4 Quantization) to simulate state-of-the-art edge deployment strategies.

---

## 📦 Dependencies

- `transformers`, `torch`, `accelerate`
- `peft`, `bitsandbytes` (for LoRA and Quantization)
- `scikit-learn`, `imbalanced-learn`
- `newsapi-python` (for OOD fetching)
- `umap-learn`, `wordcloud`, `seaborn`, `matplotlib`
- `nltk`
