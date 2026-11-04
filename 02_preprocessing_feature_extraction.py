import pandas as pd
import numpy as np
import re
import spacy
import os

print(">>> Starting Step 2: Preprocessing & Feature Extraction...")

# Load spaCy small English pipeline
try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded spaCy 'en_core_web_sm' successfully.")
except OSError:
    print("spaCy model not found. Downloading 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Define ISO/IEEE weak and ambiguous terms
WEAK_WORDS = set([
    'fast', 'faster', 'fastest', 'easy', 'easily', 'user-friendly', 'userfriendly',
    'appropriate', 'appropriately', 'soon', 'quickly', 'quick', 'robust', 'scalable',
    'efficient', 'efficiently', 'seamless', 'seamlessly', 'adequate', 'adequately',
    'minimal', 'optimum', 'flexible', 'sufficient', 'sufficiently', 'customizable',
    'reasonable', 'reasonably', 'reliable', 'instant', 'instantly', 'effective'
])

def clean_artifacts(text: str) -> str:
    """Cleans synthetic artifacts, stakeholder prefixes, and whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove Stakeholder tags like 'Stakeholder 1: Operations Manager'
    text = re.sub(r'Stakeholder\s+\d+:[^\n]+', '', text, flags=re.IGNORECASE)
    # Fix dataset synthetic token repetitions
    text = re.sub(r'\bonly\s+by\s+the\b', 'by the', text, flags=re.IGNORECASE)
    text = re.sub(r'\bonly\s+at\s+each\s+time\b', 'at a time', text, flags=re.IGNORECASE)
    text = re.sub(r'\beach\s+of\s+which\b', 'which', text, flags=re.IGNORECASE)
    # Collapse multiple whitespaces / newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Processing {len(df)} requirements with spaCy pipeline...")
    
    clean_texts = []
    word_counts = []
    char_counts = []
    vague_word_counts = []
    vague_words_found = []
    has_passive_flags = []
    has_modal_flags = []
    has_actor_flags = []
    conjunction_counts = []

    # Process in batches for high speed
    docs = list(nlp.pipe(df['cleaned_text'].tolist(), batch_size=500, disable=['ner']))

    modal_lemmas = {'shall', 'should', 'must', 'will', 'may', 'can'}

    for doc in docs:
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        lemmas = [t.lemma_.lower() for t in tokens]
        words = [t.text.lower() for t in tokens]
        
        # Word and char lengths
        word_counts.append(len(tokens))
        char_counts.append(len(doc.text))

        # 1. Ambiguous weak words
        detected_weak = [w for w in words if w in WEAK_WORDS or doc.text.lower().count(w) > 0 and w in WEAK_WORDS]
        # Check against lemmatized forms as well
        detected_weak.extend([lem for lem in lemmas if lem in WEAK_WORDS and lem not in detected_weak])
        detected_weak = list(set(detected_weak))
        vague_word_counts.append(len(detected_weak))
        vague_words_found.append(";".join(detected_weak))

        # 2. Passive voice detection (auxpass or nsubjpass)
        is_passive = any(t.dep_ in ('auxpass', 'nsubjpass') for t in doc)
        has_passive_flags.append(1 if is_passive else 0)

        # 3. Modal presence ('shall', 'should', 'must', etc.)
        has_modal = any(t.lemma_.lower() in modal_lemmas for t in doc)
        has_modal_flags.append(1 if has_modal else 0)

        # 4. Explicit actor check (subject that is not passive, e.g. nsubj)
        has_actor = any(t.dep_ == 'nsubj' for t in doc)
        has_actor_flags.append(1 if has_actor else 0)

        # 5. Conjunction count ('and', 'or' can trigger syntactic ambiguity)
        conj_count = sum(1 for t in tokens if t.pos_ == 'CCONJ' or t.text.lower() in ('and', 'or'))
        conjunction_counts.append(conj_count)

    # Attach features to dataframe
    df['char_count'] = char_counts
    df['word_count'] = word_counts
    df['vague_word_count'] = vague_word_counts
    df['vague_words'] = vague_words_found
    df['has_passive'] = has_passive_flags
    df['has_modal'] = has_modal_flags
    df['has_actor'] = has_actor_flags
    df['conjunction_count'] = conjunction_counts

    return df

def main():
    if not os.path.exists('master_requirements.csv'):
        print("[ERROR] 'master_requirements.csv' not found. Run Step 1 first!")
        return

    print("Reading master_requirements.csv...")
    df = pd.read_csv('master_requirements.csv')

    # Step 2.1: Clean text
    print("Cleaning artifacts and whitespace...")
    df['cleaned_text'] = df['requirement_text'].apply(clean_artifacts)

    # Step 2.2: Extract NLP linguistic features
    df = extract_features(df)

    # Step 2.3: Save preprocessed dataset
    output_file = 'preprocessed_requirements.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n[SUCCESS] Preprocessed dataset with engineered features saved to '{output_file}'!")
    
    # Quick sanity check print
    print("\nSample Extracted Features (First 3 rows):")
    sample_cols = ['cleaned_text', 'category', 'vague_word_count', 'vague_words', 'has_passive', 'has_actor', 'word_count']
    print(df[sample_cols].head(3).to_string())

if __name__ == '__main__':
    main()