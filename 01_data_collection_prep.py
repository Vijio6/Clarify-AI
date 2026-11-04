import pandas as pd
import numpy as np
import os

print(">>> Starting Data Preparation Script...")
print("Current working directory:", os.getcwd())

def load_and_standardize_data():
    # 1. Load dataset.csv with correct encoding
    print("Loading dataset.csv...")
    df_dataset = pd.read_csv(
        'dataset.csv', 
        encoding='latin1', 
        header=None, 
        names=['requirement_text', 'label_code']
    )
    df_dataset['source'] = 'dataset.csv'
    print(f"Loaded dataset.csv: {len(df_dataset)} rows")

    # 2. Load finalreq.csv
    print("Loading finalreq.csv...")
    df_finalreq = pd.read_csv(
        'finalreq.csv', 
        encoding='latin1', 
        header=None, 
        names=['requirement_text', 'label_code']
    )
    df_finalreq['source'] = 'finalreq.csv'
    print(f"Loaded finalreq.csv: {len(df_finalreq)} rows")

    # 3. Combine datasets
    df_combined = pd.concat([df_dataset, df_finalreq], ignore_index=True)
    print(f"Total rows before deduplication: {len(df_combined)}")

    # 4. Standardize text whitespace and remove exact duplicates
    df_combined['requirement_text'] = df_combined['requirement_text'].astype(str).str.strip()
    df_dedup = df_combined.drop_duplicates(subset=['requirement_text']).copy()
    print(f"Unique requirements retained: {len(df_dedup)}")

    # 5. Map to 4-category SRS spec and granular ambiguity types
    category_map = {
        6: 'Clear',
        3: 'Clear',
        1: 'Potentially Ambiguous',
        2: 'Potentially Ambiguous',
        4: 'Highly Ambiguous',
        5: 'Incomplete'
    }
    
    ambiguity_type_map = {
        6: 'Unambiguous / Complete',
        3: 'Semantic / Environmental Constraint',
        1: 'Syntactic / Conditional Complexity',
        2: 'Missing Actor / Open Capability',
        4: 'Compound Vagueness',
        5: 'Fragment / Missing Trigger or Action'
    }

    df_dedup['category'] = df_dedup['label_code'].map(category_map)
    df_dedup['ambiguity_type'] = df_dedup['label_code'].map(ambiguity_type_map)

    # 6. Save master unified UTF-8 dataset
    output_filename = 'master_requirements.csv'
    df_dedup.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\n[SUCCESS] Exported master dataset to '{output_filename}'")
    
    # 7. Print distribution check
    print("\n--- Master Dataset Category Distribution ---")
    print(df_dedup['category'].value_counts())
    print("\n--- Master Dataset Ambiguity Type Distribution ---")
    print(df_dedup['ambiguity_type'].value_counts())

load_and_standardize_data()