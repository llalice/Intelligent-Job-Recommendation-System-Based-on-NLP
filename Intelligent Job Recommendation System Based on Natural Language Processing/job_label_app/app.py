import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from transformers import AutoTokenizer
from utils.predict import load_model_and_tokenizer, predict_text
import json
from nltk.stem import PorterStemmer
import nltk
nltk.download("punkt")
import re

MODEL_PATH = "saved_model/bert_model.pt"
TOKENIZER_PATH = "saved_model/tokenizer_dir"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_COLS = ['label_skill', 'label_language', 'label_experience', 'label_benefit', 'label_warning']

LABEL_DISPLAY = {
    "label_skill": "Skill-based",
    "label_language": "Language Requirement",
    "label_experience": "Experience Required",
    "label_benefit": "Comes with Benefits",
    "label_warning": "May Have Stress Factors"
}

@st.cache_resource
def load_keywords():
    with open("saved_model/expanded_keywords.json") as f:
        return json.load(f)

expanded_keywords = load_keywords()

@st.cache_resource
def get_model():
    return load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH, LABEL_COLS, DEVICE)

model, tokenizer = get_model()

st.set_page_config(page_title="Find Your Best Job", layout="wide")
st.title("🚀 Smart Job Finder for Career Seekers")

st.sidebar.header("📄 Upload Job Listings File")
uploaded_file = st.sidebar.file_uploader("Upload job data (.csv or .xlsx)", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("📌 Try Predict From Your Own Text")
user_input = st.sidebar.text_area("Paste your resume or job description here")

if st.sidebar.button("🔍 Analyze Description"):
    if user_input.strip():
        try:
            model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH, LABEL_COLS, DEVICE)
            with open("saved_model/expanded_keywords.json") as f:
                expanded_keywords = json.load(f)

            pred_df, prob_df = predict_text([user_input], model, tokenizer, LABEL_COLS, DEVICE)
            predicted = pred_df.iloc[0].to_dict()
            probs = prob_df.iloc[0].to_dict()

            st.markdown("## ✅ Prediction Results")
            for label, value in predicted.items():
                if value == 1:
                    score = probs[f"prob_{label}"]
                    st.markdown(f"- **{LABEL_DISPLAY.get(label, label)}** (Probability: `{score:.2f}`)")

            st.markdown("### 🧩 Matched Keywords by Category")
            stemmer = PorterStemmer()

            def keyword_match(text, keyword_set):
                text_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
                tokens = [stemmer.stem(w) for w in text_clean.split() if len(w) > 2]
                joined = " ".join(tokens)
                matches = []
                for kw in keyword_set:
                    kw_tokens = [stemmer.stem(w) for w in kw.lower().split() if len(w) > 2]
                    if all(t in joined for t in kw_tokens):
                        matches.append(kw)
                return matches

            for label, value in predicted.items():
                if value == 1:
                    cat = label.replace("label_", "")
                    hits = keyword_match(user_input, expanded_keywords.get(cat, []))
                    if hits:
                        st.markdown(f"**{LABEL_DISPLAY.get(label, label)}** → `{', '.join(hits)}`")

        except Exception as e:
            st.warning(f"Model or keyword loading error: {e}")
    else:
        st.warning("Please enter some text to analyze.")

if uploaded_file:
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding="latin1")
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to read the uploaded file: {e}")
        st.stop()

    desc_col = None
    for col in df.columns:
        if "description" in col.lower():
            desc_col = col
            break
    if not desc_col:
        st.error("No description column found.")
        st.stop()

    pred_df, prob_df = predict_text(df[desc_col].astype(str).tolist(), model, tokenizer, LABEL_COLS, DEVICE)
    df = pd.concat([df.reset_index(drop=True), pred_df, prob_df], axis=1)

    def keyword_match(text, keyword_set):
        stemmer = PorterStemmer()
        text_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = [stemmer.stem(w) for w in text_clean.split() if len(w) > 2]
        joined = " ".join(tokens)
        matches = []
        for kw in keyword_set:
            kw_tokens = [stemmer.stem(w) for w in kw.lower().split() if len(w) > 2]
            if all(t in joined for t in kw_tokens):
                matches.append(kw)
        return matches

    for label in LABEL_COLS:
        cat = label.replace("label_", "")
        df[f"matched_{cat}"] = df[desc_col].astype(str).apply(lambda x: keyword_match(x, expanded_keywords.get(cat, [])))

    st.markdown("## 🔍 Data Overview")
    st.dataframe(df.head())

    st.markdown("---")
    st.markdown("### 📊 In-depth Analysis")

    st.markdown("#### ✅ Summary by Label Category")
    label_summary = df[LABEL_COLS].sum().rename(lambda x: LABEL_DISPLAY.get(x, x))
    st.bar_chart(label_summary)

    st.markdown("#### 🏆 Top Companies by Label Type")
    for label in LABEL_COLS:
        top_comp = df[df[label] == 1].groupby("company")["monthly_avg_rm"].mean().sort_values(ascending=False).head(5)
        if not top_comp.empty:
            st.markdown(f"**Top Companies for {LABEL_DISPLAY[label]}**")
            st.dataframe(top_comp.reset_index())

    st.markdown("#### 💰 Top Salary Companies (Overall)")
    top_salary = df.groupby("company")["monthly_avg_rm"].mean().sort_values(ascending=False).head(5)
    st.dataframe(top_salary.reset_index().rename(columns={"monthly_avg_rm": "Avg Monthly Salary (RM)"}))

    st.markdown("#### 💡 Filter Label Contents by Company")
    company_to_check = st.selectbox("Select Company to View Label Details", df["company"].dropna().unique())
    company_df = df[df["company"] == company_to_check]
    for label in LABEL_COLS:
        values = company_df[company_df[label] == 1][f"matched_{label.replace('label_', '')}"]
        all_keywords = [kw for sublist in values for kw in sublist]
        if all_keywords:
            st.markdown(f"**{LABEL_DISPLAY[label]}**: {', '.join(set(all_keywords))}")

    st.markdown("#### 🔍 Keyword-Specific Job Filter")
    label_options = list(LABEL_DISPLAY.values())
    selected_label_display = st.selectbox("Select Label Type to Filter Jobs By Keywords", label_options)
    selected_label = [k for k, v in LABEL_DISPLAY.items() if v == selected_label_display][0]
    matched_col = f"matched_{selected_label.replace('label_', '')}"

    if matched_col in df.columns:
        keyword_expanded = df[[matched_col, "job_title", "company", "monthly_avg_rm"]].explode(matched_col)
        keyword_expanded = keyword_expanded.dropna(subset=[matched_col])
        keyword_expanded = keyword_expanded.rename(columns={matched_col: "keyword"})
        if not keyword_expanded.empty:
            st.dataframe(keyword_expanded.sort_values("keyword"))

    st.markdown("#### 🗺️ Location-Based Salary Insight")
    if 'location' in df.columns:
        location_avg = df.groupby("location")["monthly_avg_rm"].mean().sort_values(ascending=False)
        st.bar_chart(location_avg)

    st.download_button("📥 Download Full Labeled Data", data=df.to_csv(index=False).encode(), file_name="labeled_jobs.csv")
else:
    st.info("👆 Upload a job dataset to begin your exploration.")
