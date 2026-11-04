import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

print(">>> Starting Steps 4, 5, and 6: Model Training, Evaluation & Comparison...")

output_dir = "model_results"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Preprocessed Data
df = pd.read_csv("preprocessed_requirements.csv")
print(f"Loaded dataset: {df.shape[0]} rows")

# Drop any rows with NaN in critical columns
df = df.dropna(subset=['cleaned_text', 'category']).copy()

# 2. Features and Target
X = df[['cleaned_text', 'vague_word_count', 'has_passive', 'has_actor', 'has_modal', 'word_count', 'conjunction_count']]
y_raw = df['category']

# Encode target classes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_
print(f"Target classes ({len(classes)}): {list(classes)}")

# 3. Step 4: 80% Train / 20% Test Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} samples (80%)")
print(f"Testing set:  {X_test.shape[0]} samples (20%)")

# 4. Feature Extraction Preprocessor
text_features = 'cleaned_text'
numeric_features = ['vague_word_count', 'has_passive', 'has_actor', 'has_modal', 'word_count', 'conjunction_count']

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True), text_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# 5. Define Candidate Models
# Note: LinearSVC is calibrated with CalibratedClassifierCV to enable predict_proba for ROC-AUC and scoring
models = {
    "Multinomial Naive Bayes": Pipeline([
        ('prep', ColumnTransformer(transformers=[
            ('text', TfidfVectorizer(ngram_range=(1, 2), max_features=5000), text_features)
        ])),
        ('clf', MultinomialNB())
    ]),
    "Logistic Regression": Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    "Linear Support Vector Machine": Pipeline([
        ('prep', preprocessor),
        ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)))
    ]),
    "Random Forest": Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1))
    ]),
    "XGBoost Classifier": Pipeline([
        ('prep', preprocessor),
        ('clf', XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=42, n_jobs=-1))
    ])
}

# 6. Step 5 & 6: Train, Evaluate, and Compare
results = []

for name, pipeline in models.items():
    print(f"\n--- Training: {name} ---")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    # Probability prediction for ROC-AUC
    try:
        y_proba = pipeline.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc_score = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy:       {acc*100:.2f}%")
    print(f"  Macro F1:       {f1_macro:.4f}")
    print(f"  ROC-AUC (OvR):  {auc_score:.4f}")

    results.append({
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Macro Precision": round(prec_macro, 4),
        "Macro Recall": round(rec_macro, 4),
        "Macro F1": round(f1_macro, 4),
        "Weighted F1": round(f1_weighted, 4),
        "ROC AUC (OvR)": round(auc_score, 4) if not np.isnan(auc_score) else "N/A",
        "pipeline": pipeline
    })

# Convert results to DataFrame
comparison_df = pd.DataFrame(results)

# 7. Step 6: Model Comparison & Best Model Selection
comparison_table = comparison_df.drop(columns=['pipeline']).sort_values(by="Macro F1", ascending=False)
print("\n" + "="*80)
print("                       MODEL COMPARISON LEADERBOARD")
print("="*80)
print(comparison_table.to_string(index=False))

# Export leaderboard to CSV
comparison_table.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

# Identify Best Model by Macro F1
best_model_row = comparison_df.sort_values(by="Macro F1", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]
best_pipeline = best_model_row["pipeline"]
print(f"\n[WINNING MODEL] => '{best_model_name}' achieved highest Macro F1: {best_model_row['Macro F1']}")

# Save the Best Model Pipeline and Label Encoder
joblib.dump(best_pipeline, os.path.join(output_dir, "best_model.joblib"))
joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
print(f"Saved best model pipeline to '{output_dir}/best_model.joblib'")
print(f"Saved label encoder to '{output_dir}/label_encoder.joblib'")

# 8. Generate Visual Confusion Matrix for the Winning Model
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title(f"Confusion Matrix: {best_model_name}", fontsize=13, weight='bold', pad=15)
plt.xlabel("Predicted Category", fontsize=11, weight='bold')
plt.ylabel("Actual Category", fontsize=11, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_model_confusion_matrix.png"), dpi=300)
plt.close()
print(f"Saved confusion matrix chart to '{output_dir}/best_model_confusion_matrix.png'")

# 9. Comparative Performance Bar Chart
plt.figure(figsize=(10, 5))
plot_df = comparison_df.sort_values(by="Macro F1", ascending=True)
bars = plt.barh(plot_df['Model'], plot_df['Macro F1'], color="#2b5c8f")
plt.title("Model Comparison: Macro F1 Score (Held-Out 20% Test Set)", fontsize=13, weight='bold', pad=15)
plt.xlabel("Macro F1 Score", fontsize=11, weight='bold')
plt.xlim(0, 1.0)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2., f'{width:.4f}',
             va='center', ha='left', fontsize=10, weight='semibold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_f1_comparison.png"), dpi=300)
plt.close()
print(f"Saved model comparison plot to '{output_dir}/model_f1_comparison.png'")

print("\n[SUCCESS] Steps 4, 5, and 6 completed successfully!")