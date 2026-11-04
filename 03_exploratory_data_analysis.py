import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(">>> Starting Step 3: Exploratory Data Analysis (EDA)...")

# Setup aesthetic style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# Create output folder for charts
output_dir = "eda_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Preprocessed Data
df = pd.read_csv("preprocessed_requirements.csv")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------------------------------------
# PLOT 1: Target SRS Category Distribution
# -------------------------------------------------------------
plt.figure(figsize=(9, 5))
cat_counts = df['category'].value_counts()
ax = sns.barplot(x=cat_counts.index, y=cat_counts.values, palette="Blues_r")
plt.title("Distribution of Requirements by SRS Ambiguity Category", fontsize=14, weight='bold', pad=15)
plt.xlabel("Category", fontsize=11, weight='bold')
plt.ylabel("Requirement Count", fontsize=11, weight='bold')

for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f'{height} ({height/len(df)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=10, weight='semibold',
                xytext=(0, 4), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=300)
plt.close()
print("Saved: 01_class_distribution.png")

# -------------------------------------------------------------
# PLOT 2: Ambiguity Subtypes Distribution
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
type_counts = df['ambiguity_type'].value_counts()
ax = sns.barplot(x=type_counts.values, y=type_counts.index, palette="viridis")
plt.title("Distribution of Granular Ambiguity Types", fontsize=14, weight='bold', pad=15)
plt.xlabel("Count", fontsize=11, weight='bold')
plt.ylabel("Ambiguity Type", fontsize=11, weight='bold')

for p in ax.patches:
    width = int(p.get_width())
    ax.annotate(f'{width}',
                (width, p.get_y() + p.get_height() / 2.),
                ha='left', va='center', fontsize=10, weight='semibold',
                xytext=(6, 0), textcoords='offset points')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_ambiguity_types.png"), dpi=300)
plt.close()
print("Saved: 02_ambiguity_types.png")

# -------------------------------------------------------------
# PLOT 3: Requirement Length (Word Count) by Category
# -------------------------------------------------------------
plt.figure(figsize=(9, 5))
# Clip extreme outliers at 99th percentile for clean boxplot view
upper_limit = np.percentile(df['word_count'], 99)
sns.boxplot(
    data=df[df['word_count'] <= upper_limit],
    x='category',
    y='word_count',
    palette="Set2"
)
plt.title("Requirement Length (Word Count) Across Ambiguity Categories", fontsize=14, weight='bold', pad=15)
plt.xlabel("Category", fontsize=11, weight='bold')
plt.ylabel("Word Count", fontsize=11, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_word_count_by_category.png"), dpi=300)
plt.close()
print("Saved: 03_word_count_by_category.png")

# -------------------------------------------------------------
# PLOT 4: Linguistic Feature Analysis (Passive Voice & Explicit Actor)
# -------------------------------------------------------------
prop_df = df.groupby('category')[['has_passive', 'has_actor']].mean() * 100
prop_df = prop_df.reset_index().melt(id_vars='category', value_vars=['has_passive', 'has_actor'],
                                     var_name='Feature', value_name='Percentage')
prop_df['Feature'] = prop_df['Feature'].map({'has_passive': 'Uses Passive Voice', 'has_actor': 'Has Explicit Actor'})

plt.figure(figsize=(9, 5))
ax = sns.barplot(data=prop_df, x='category', y='Percentage', hue='Feature', palette="coolwarm")
plt.title("Linguistic Indicators by Requirement Category (%)", fontsize=14, weight='bold', pad=15)
plt.xlabel("Category", fontsize=11, weight='bold')
plt.ylabel("Percentage of Requirements (%)", fontsize=11, weight='bold')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "04_passive_voice_vs_actors.png"), dpi=300)
plt.close()
print("Saved: 04_passive_voice_vs_actors.png")

# -------------------------------------------------------------
# PLOT 5: Top 15 Vague / Weak Trigger Words
# -------------------------------------------------------------
all_vague_words = []
df['vague_words'].dropna().apply(
    lambda x: all_vague_words.extend([w.strip() for w in str(x).split(';') if w.strip()])
)

if all_vague_words:
    vague_series = pd.Series(all_vague_words).value_counts().head(15)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=vague_series.values, y=vague_series.index, palette="mako")
    plt.title("Top 15 Detected Ambiguous / Weak Keywords", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Frequency", fontsize=11, weight='bold')
    plt.ylabel("Keyword", fontsize=11, weight='bold')

    for p in ax.patches:
        width = int(p.get_width())
        ax.annotate(f'{width}',
                    (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', fontsize=9,
                    xytext=(5, 0), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_top_vague_words.png"), dpi=300)
    plt.close()
    print("Saved: 05_top_vague_words.png")

print(f"\n[SUCCESS] All EDA visualizations exported to the '{output_dir}/' folder!")