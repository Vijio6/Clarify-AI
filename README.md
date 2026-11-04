# Clarify AI: AI-Powered SRS Ambiguity Detector & IEEE-830 Restructurer

Clarify AI is a production-grade NLP and Machine Learning auditing suite designed to detect ambiguity, vague terminology, grammatical deficiencies, and semantic contradictions in Software Requirements Specifications (SRS). Compliant with **IEEE-830** and **ISO-29148** requirements engineering standards, the system pairs classical ML classification with dynamic LLaMA streaming rewrites.

---

## Key Features

- **Single Requirement Inspector:** Computes an Ambiguity Score (0–100), severity tier, missing actor flags, and passive voice detection.
- **Dynamic IEEE-830 Rewrites:** Real-time token streaming via Groq LLaMA models providing automated explanations, missing quantitative constraints, and testable rewrites.
- **Batch Document Analysis:** High-throughput requirement evaluation with parallelized spaCy tokenization (`nlp.pipe`) and instant CSV export.
- **Semantic Conflict & Duplicate Detector:** Pairwise cosine similarity and polarity inversion checks to detect contradictory or duplicate functional statements.
- **EDA & Diagnostic Dashboard:** Live leaderboard, confusion matrix visualizer, and dataset distributions.

---

## System Architecture & Pipeline

1. **Data Curation & Deduplication:** Unified 7,546 unique requirement statements into standardized UTF-8 encodings across 4 core classes: `Clear`, `Potentially Ambiguous`, `Highly Ambiguous`, and `Incomplete`.
2. **NLP Feature Engineering:** Extracted lexicons of IEEE weak words, grammatical dependency structures (passive voice, syntactic actor subjects, conjunction counts), and token lengths using spaCy (`en_core_web_sm`).
3. **Model Selection:** Stratified 80/20 train/test evaluation across 5 candidate classifiers (Logistic Regression, Support Vector Classifier, Random Forest, Multinomial Naive Bayes, and XGBoost). The winning pipeline combines TF-IDF n-grams with scaled linguistic features.
4. **Interactive Dashboard:** Built with Streamlit, cached model pipelines, and dynamic API discovery.

---

## Model Benchmark Results

Evaluated on a 20% held-out stratified test set:

| Model | Macro F1 | Weighted F1 | Test Accuracy |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.5789** | **0.6975** | **70.2%** |
| Random Forest | 0.5412 | 0.6720 | 68.4% |
| Support Vector Classifier (SVC) | 0.5398 | 0.6705 | 68.1% |
| XGBoost Classifier | 0.5244 | 0.6612 | 67.5% |
| Multinomial Naive Bayes | 0.4410 | 0.6120 | 63.8% |

---

## Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Vijio6/Clarify-AI.git](https://github.com/Vijio6/Clarify-AI.git)
   cd Clarify-AI
