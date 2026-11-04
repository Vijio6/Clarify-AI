import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(
    page_title="Clarify AI - SRS Ambiguity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Standard IEEE / ISO weak words & multi-word expressions
WEAK_WORDS = {
    'fast', 'faster', 'fastest', 'easy', 'easily', 'user-friendly', 'user friendly',
    'userfriendly', 'appropriate', 'appropriately', 'soon', 'quickly', 'quick',
    'robust', 'scalable', 'efficient', 'efficiently', 'seamless', 'seamlessly',
    'adequate', 'adequately', 'minimal', 'optimum', 'flexible', 'sufficient',
    'sufficiently', 'customizable', 'reasonable', 'reasonably', 'reliable',
    'instant', 'instantly', 'effective', 'state-of-the-art', 'state of the art'
}
MODAL_WORDS = {'shall', 'should', 'must', 'will', 'may', 'can'}

@st.cache_resource
def load_resources():
    model_path = os.path.join("model_results", "best_model.joblib")
    encoder_path = os.path.join("model_results", "label_encoder.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None, None
        
    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    
    # Clean loader without scoping conflicts
    try:
        import en_core_web_sm
        nlp_model = en_core_web_sm.load()
    except Exception:
        try:
            nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp_model = spacy.load("en_core_web_sm")
        
    return pipeline, label_encoder, nlp_model

pipeline, label_encoder, nlp = load_resources()

def clean_artifacts(text: str) -> str:
    text = re.sub(r'Stakeholder\s+\d+:[^\n]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bonly\s+by\s+the\b', 'by the', text, flags=re.IGNORECASE)
    text = re.sub(r'\bonly\s+at\s+each\s+time\b', 'at a time', text, flags=re.IGNORECASE)
    text = re.sub(r'\beach\s+of\s+which\b', 'which', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def find_weak_words(raw_text: str, doc) -> list:
    """Detects weak words across both token lemmas and multi-word phrase regex."""
    found = set()
    text_lower = raw_text.lower()
    
    for w in WEAK_WORDS:
        pattern = rf'\b{re.escape(w)}\b'
        if re.search(pattern, text_lower):
            found.add(w)
            
    for t in doc:
        if not t.is_punct and not t.is_space:
            if t.lemma_.lower() in WEAK_WORDS:
                found.add(t.lemma_.lower())
            if t.text.lower() in WEAK_WORDS:
                found.add(t.text.lower())
                
    return sorted(list(found))

def extract_single_features(text: str):
    cleaned = clean_artifacts(text)
    doc = nlp(cleaned)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]

    detected_weak = find_weak_words(text, doc)
    is_passive = any(t.dep_ in ('auxpass', 'nsubjpass') for t in doc)
    has_modal = any(t.lemma_.lower() in MODAL_WORDS for t in doc)
    has_actor = any(t.dep_ == 'nsubj' for t in doc)
    conj_count = sum(1 for t in tokens if t.pos_ == 'CCONJ' or t.text.lower() in ('and', 'or'))

    feat_df = pd.DataFrame([{
        'cleaned_text': cleaned,
        'vague_word_count': len(detected_weak),
        'has_passive': 1 if is_passive else 0,
        'has_actor': 1 if has_actor else 0,
        'has_modal': 1 if has_modal else 0,
        'word_count': len(tokens),
        'conjunction_count': conj_count
    }])
    return feat_df, cleaned, detected_weak, is_passive, has_actor

def get_groq_stream(api_key: str, text: str, category: str, weak_words: list):
    """Dynamically finds available text models on the account and streams output without cutoff."""
    if not api_key or not GROQ_AVAILABLE:
        return None
    try:
        client = Groq(api_key=api_key)
        prompt = f"""You are a senior Software Requirements Engineer specializing in IEEE-830 and ISO-29148 standards.
Analyze this requirement:
Requirement: "{text}"
Category: {category}
Detected Ambiguous Words: {', '.join(weak_words) if weak_words else 'None'}

Provide a structured response:
### 1. Ambiguity Explanation
Explain in 2 concise sentences why this statement is ambiguous, unmeasurable, or incomplete.

### 2. Missing Context / Testable Metrics
List bullet points of exact missing criteria (e.g., maximum latency in ms, error rates, actor roles).

### 3. IEEE-830 Compliant Rewrite
Provide a complete, verifiable rewrite starting with 'The system shall...'."""

        # Fetch active models available on this API key
        try:
            models_data = client.models.list().data
            active_ids = [m.id for m in models_data]
        except Exception:
            active_ids = []

        chat_models = [m for m in active_ids if not any(x in m.lower() for x in ['whisper', 'guard', 'safeguard'])]

        preferred_order = [
            "llama3-70b-8192",
            "llama3-8b-8192",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
            "llama-3.2-3b-preview",
            "llama-3.2-1b-preview"
        ]

        sorted_candidates = [m for m in preferred_order if m in chat_models]
        for m in chat_models:
            if m not in sorted_candidates:
                sorted_candidates.append(m)

        if not sorted_candidates:
            sorted_candidates = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]

        last_error = None
        for model_id in sorted_candidates:
            try:
                stream = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_id,
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
                return
            except Exception as err:
                last_error = err
                continue

        yield f"LLM Generation Error: {str(last_error)}"
    except Exception as e:
        yield f"LLM Generation Error: {str(e)}"

# Sidebar Navigation
st.sidebar.title("Clarify AI")
st.sidebar.caption("AI-Based Software Requirement Ambiguity Detector")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Requirement Inspector", "Batch Document Analysis", "Semantic Conflict Detector", "Model Performance & EDA"]
)

st.sidebar.divider()
st.sidebar.subheader("Groq API Key")

# Seamless fallback to Streamlit Cloud secrets
default_api_key = ""
try:
    if "GROQ_API_KEY" in st.secrets:
        default_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key (Optional)", 
    value=default_api_key,
    type="password", 
    help="Enables dynamic IEEE-830 rewrites, ambiguity explanations, and missing context prompts."
)

if pipeline is None:
    st.error("Model files not found in `model_results/`. Please verify `model_results/` contains `best_model.joblib` and `label_encoder.joblib`.")
    st.stop()

# -------------------------------------------------------------
# MODE 1: REQUIREMENT INSPECTOR
# -------------------------------------------------------------
if app_mode == "Requirement Inspector":
    st.title("Requirement Ambiguity Inspector")
    st.write("Analyze individual software requirements against IEEE-830 ambiguity benchmarks.")

    sample_reqs = [
        "The user interface must be user-friendly and respond quickly.",
        "The system should provide fast and easy access to reports.",
        "Ability to quickly scale applications up or down based on demand.",
        "The system shall calculate tax for the order within 500ms of checkout submission.",
        "Visualize time-series data without errors."
    ]
    selected_sample = st.selectbox("Load sample requirement:", ["-- Select or type below --"] + sample_reqs)

    input_text = st.text_area(
        "Enter Requirement Statement:",
        value=selected_sample if selected_sample != "-- Select or type below --" else "",
        placeholder="e.g. The user interface must be user-friendly and respond quickly."
    )

    if st.button("Analyze Requirement", type="primary"):
        if not input_text.strip():
            st.warning("Please enter a requirement statement.")
        else:
            feat_df, cleaned, weak_words, is_passive, has_actor = extract_single_features(input_text)
            pred_idx = pipeline.predict(feat_df)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            probas = pipeline.predict_proba(feat_df)[0]

            clear_idx = np.where(label_encoder.classes_ == "Clear")[0]
            p_clear = probas[clear_idx[0]] if len(clear_idx) > 0 else 0.0
            raw_score = (1.0 - p_clear) * 100
            ambiguity_score = int(min(100, max(5, raw_score + (len(weak_words) * 10) + (10 if is_passive else 0))))

            color_theme = {
                "Clear": ("#22c55e", "rgba(34, 197, 94, 0.15)", "#4ade80"),
                "Potentially Ambiguous": ("#eab308", "rgba(234, 179, 8, 0.15)", "#facc15"),
                "Highly Ambiguous": ("#ef4444", "rgba(239, 68, 68, 0.15)", "#f87171"),
                "Incomplete": ("#f97316", "rgba(249, 115, 22, 0.15)", "#fb923c")
            }
            border_c, bg_c, text_c = color_theme.get(pred_label, ("#3b82f6", "rgba(59, 130, 246, 0.15)", "#60a5fa"))

            st.markdown(
                f"""
                <div style="background-color: {bg_c}; border: 1px solid {border_c}; border-left: 8px solid {border_c}; padding: 14px 20px; border-radius: 8px; margin: 15px 0 25px 0;">
                    <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;">Classification Verdict</div>
                    <div style="font-size: 24px; font-weight: 800; color: {text_c}; margin-top: 4px;">{pred_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Ambiguity Score", f"{ambiguity_score} / 100")
            col2.metric("Weak Words Found", len(weak_words))
            col3.metric("Actor Present", "Yes" if has_actor else "Missing")

            st.divider()

            st.subheader("Linguistic Diagnostics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Detected Weak/Ambiguous Terms:**")
                if weak_words:
                    st.write(" ".join([f"`{w}`" for w in weak_words]))
                else:
                    st.success("No vague trigger terms detected.")
            with col_b:
                st.write("**Grammatical Structure:**")
                st.write(f"- Passive Voice: `{'Detected' if is_passive else 'None'}`")
                st.write(f"- Explicit Actor: `{'Present' if has_actor else 'Missing'}`")
                st.write(f"- Word Count: `{feat_df['word_count'].iloc[0]}` tokens")

            st.subheader("Requirement Improvement & Rewrite")
            if groq_api_key:
                stream_generator = get_groq_stream(groq_api_key, input_text, pred_label, weak_words)
                if stream_generator:
                    st.write_stream(stream_generator)
            else:
                rule_rewrite = input_text.strip()
                if not has_actor:
                    rule_rewrite = "The system shall " + rule_rewrite[0].lower() + rule_rewrite[1:]
                for w in weak_words:
                    rule_rewrite = re.sub(rf'\b{re.escape(w)}\b', f'[{w}: specify measurable metric]', rule_rewrite, flags=re.IGNORECASE)
                if is_passive:
                    rule_rewrite += " (Specify the responsible actor/role)."
                st.info(rule_rewrite)
                st.caption("Tip: Add a Groq API key in the sidebar for real-time explanations, missing context points, and full rewrites.")

# -------------------------------------------------------------
# MODE 2: BATCH DOCUMENT ANALYSIS
# -------------------------------------------------------------
elif app_mode == "Batch Document Analysis":
    st.title("Batch Document Analysis")
    st.write("Upload a requirements specification document (CSV or TXT) to evaluate full documents.")

    uploaded_file = st.file_uploader("Upload requirements file", type=["csv", "txt"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            doc_df = pd.read_csv(uploaded_file)
            req_col = st.selectbox("Select requirement column:", doc_df.columns)
            req_list = doc_df[req_col].dropna().astype(str).tolist()
        else:
            content = uploaded_file.read().decode("utf-8")
            req_list = [line.strip() for line in content.split("\n") if line.strip()]

        total_count = len(req_list)
        st.write(f"Loaded **{total_count}** requirement statements.")

        col_opt1, col_opt2 = st.columns([1, 2])
        with col_opt1:
            run_sample = st.checkbox("Quick Audit (first N items)", value=(total_count > 500))
        with col_opt2:
            sample_size = st.number_input("Sample size:", min_value=10, max_value=total_count, value=min(200, total_count)) if run_sample else total_count

        active_reqs = req_list[:sample_size] if run_sample else req_list

        if st.button("Run Batch Ambiguity Audit", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            status_text.text("Cleaning and tokenizing requirements in batches...")

            cleaned_texts = [clean_artifacts(r) for r in active_reqs]

            # Vectorized spaCy processing
            nlp_docs = list(nlp.pipe(cleaned_texts, batch_size=256, n_process=1))

            records = []
            feature_rows = []
            total_items = len(active_reqs)

            for idx, (raw_req, cleaned, doc) in enumerate(zip(active_reqs, cleaned_texts, nlp_docs)):
                tokens = [t for t in doc if not t.is_punct and not t.is_space]
                detected_weak = find_weak_words(raw_req, doc)
                is_passive = any(t.dep_ in ('auxpass', 'nsubjpass') for t in doc)
                has_modal = any(t.lemma_.lower() in MODAL_WORDS for t in doc)
                has_actor = any(t.dep_ == 'nsubj' for t in doc)
                conj_count = sum(1 for t in tokens if t.pos_ == 'CCONJ' or t.text.lower() in ('and', 'or'))

                feature_rows.append({
                    'cleaned_text': cleaned,
                    'vague_word_count': len(detected_weak),
                    'has_passive': 1 if is_passive else 0,
                    'has_actor': 1 if has_actor else 0,
                    'has_modal': 1 if has_modal else 0,
                    'word_count': len(tokens),
                    'conjunction_count': conj_count
                })

                records.append({
                    "Requirement": raw_req,
                    "Weak Words": "; ".join(detected_weak) if detected_weak else "None",
                    "Passive Voice": "Yes" if is_passive else "No",
                    "_weak_count": len(detected_weak),
                    "_is_passive": is_passive
                })

                if (idx + 1) % max(1, total_items // 20) == 0:
                    progress_bar.progress((idx + 1) / total_items)
                    status_text.text(f"Processed {idx + 1}/{total_items} requirements...")

            status_text.text("Running model predictions...")
            batch_feat_df = pd.DataFrame(feature_rows)
            pred_indices = pipeline.predict(batch_feat_df)
            pred_labels = label_encoder.inverse_transform(pred_indices)
            pred_probas = pipeline.predict_proba(batch_feat_df)

            clear_idx = np.where(label_encoder.classes_ == "Clear")[0]
            clear_col = clear_idx[0] if len(clear_idx) > 0 else 0

            for i in range(total_items):
                p_clear = pred_probas[i][clear_col]
                raw_score = (1.0 - p_clear) * 100
                ambiguity_score = int(min(100, max(5, raw_score + (records[i]["_weak_count"] * 10) + (10 if records[i]["_is_passive"] else 0))))
                records[i]["Category"] = pred_labels[i]
                records[i]["Ambiguity Score"] = ambiguity_score
                del records[i]["_weak_count"]
                del records[i]["_is_passive"]

            progress_bar.progress(1.0)
            status_text.empty()
            progress_bar.empty()

            res_df = pd.DataFrame(records)[["Requirement", "Category", "Ambiguity Score", "Weak Words", "Passive Voice"]]

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Analyzed", len(res_df))
            kpi2.metric("Clear Requirements", len(res_df[res_df["Category"] == "Clear"]))
            kpi3.metric("Potentially Ambiguous", len(res_df[res_df["Category"] == "Potentially Ambiguous"]))
            kpi4.metric("High Ambiguity / Incomplete", len(res_df[res_df["Category"].isin(["Highly Ambiguous", "Incomplete"])]))

            st.dataframe(res_df, use_container_width=True)

            csv_download = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Analysis Report (CSV)",
                data=csv_download,
                file_name="clarify_ai_batch_analysis.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------
# MODE 3: SEMANTIC CONFLICT DETECTOR
# -------------------------------------------------------------
elif app_mode == "Semantic Conflict Detector":
    st.title("Semantic Conflict & Redundancy Detector")
    st.write("Identifies duplicate requirements and contradictory statements.")

    default_batch = (
        "The system shall allow any user to download data without authentication.\n"
        "Unregistered users cannot download any system reports under any circumstances.\n"
        "Ability to quickly scale applications up or down based on demand.\n"
        "Ability to scale applications based on demand."
    )
    batch_input = st.text_area("Enter requirements (one per line):", value=default_batch, height=180)

    if st.button("Check Consistency & Conflicts", type="primary"):
        lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        if len(lines) < 2:
            st.warning("Please provide at least two requirements to compare.")
        else:
            tfidf = pipeline.named_steps['prep'].named_transformers_['text']
            vecs = tfidf.transform(lines)
            sim_matrix = cosine_similarity(vecs)

            conflict_pairs = []
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    sim = sim_matrix[i][j]
                    if sim > 0.45:
                        req_a, req_b = lines[i], lines[j]
                        has_negation_a = bool(re.search(r'\b(not|cannot|no|without|never)\b', req_a, re.I))
                        has_negation_b = bool(re.search(r'\b(not|cannot|no|without|never)\b', req_b, re.I))
                        
                        issue_type = "Potential Conflict" if (has_negation_a != has_negation_b) else "Potential Duplicate / Redundancy"
                        conflict_pairs.append({
                            "Requirement A": req_a,
                            "Requirement B": req_b,
                            "Similarity": f"{sim * 100:.1f}%",
                            "Issue Type": issue_type
                        })

            if conflict_pairs:
                st.subheader(f"Detected Issues ({len(conflict_pairs)})")
                st.table(pd.DataFrame(conflict_pairs))
            else:
                st.success("No significant semantic conflicts or duplicate pairs detected.")

# -------------------------------------------------------------
# MODE 4: MODEL PERFORMANCE & EDA
# -------------------------------------------------------------
elif app_mode == "Model Performance & EDA":
    st.title("Model Benchmarks & Dataset EDA")

    tab1, tab2 = st.tabs(["Model Comparison Leaderboard", "Exploratory Data Analysis"])

    with tab1:
        st.subheader("Model Evaluation on Held-Out Test Set (20%)")
        comp_csv = os.path.join("model_results", "model_comparison.csv")
        if os.path.exists(comp_csv):
            st.dataframe(pd.read_csv(comp_csv), use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        cm_path = os.path.join("model_results", "best_model_confusion_matrix.png")
        f1_path = os.path.join("model_results", "model_f1_comparison.png")
        if os.path.exists(cm_path):
            col_m1.image(cm_path, caption="Winning Model Confusion Matrix")
        if os.path.exists(f1_path):
            col_m2.image(f1_path, caption="F1 Score Comparison")

    with tab2:
        st.subheader("Exploratory Visualizations")
        eda_folder = "eda_plots"
        if os.path.exists(eda_folder):
            eda_files = sorted([f for f in os.listdir(eda_folder) if f.endswith(".png")])
            for f in eda_files:
                st.image(os.path.join(eda_folder, f), caption=f.replace("_", " ").replace(".png", "").title())
        else:
            st.info("EDA plots not found. Run Step 3 to generate them.")