# ====================================================
# 📌 Mobile Reviews Sentiment Analyzer (Full App)
# ====================================================

import pandas as pd
import numpy as np
import re
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

# ====================================================
# 1️⃣ Streamlit Page Config
# ====================================================
st.set_page_config(
    page_title="Mobile Reviews Sentiment Analyzer",
    layout="wide"
)

st.title("📱 Mobile Reviews Sentiment Analysis")
st.markdown("""
This app automatically generates **pseudo-labels** using **VADER**, trains a **Naive Bayes ML model**, 
and evaluates its performance internally. All done **without needing labeled CSVs**.
""")

# ====================================================
# 2️⃣ Load Dataset
# ====================================================
st.subheader("Step 1: Load Dataset")
uploaded_file = st.file_uploader("Mobile_reviews.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=["body"])
    st.success(f"Dataset loaded: {df.shape[0]} rows")
    
    # Show sample
    st.write(df.head())

    # ====================================================
    # 3️⃣ Text Cleaning
    # ====================================================
    st.subheader("Step 2: Text Cleaning")
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        return text.strip()

    df["clean_body"] = df["body"].apply(clean_text)
    st.write("Sample cleaned text:")
    st.write(df[["body", "clean_body"]].head())

    # ====================================================
    # 4️⃣ Generate Pseudo Labels (VADER)
    # ====================================================
  
    sia = SentimentIntensityAnalyzer()

    def vader_label(text):
        score = sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return None

    df["true_sentiment"] = df["clean_body"].apply(vader_label)
    df = df.dropna(subset=["true_sentiment"])
    st.write("Label distribution:")
    st.bar_chart(df["true_sentiment"].value_counts())

    # ====================================================
    # 5️⃣ Train Naive Bayes Model
    # ====================================================
  
    X = df["clean_body"]
    y_true = df["true_sentiment"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_true, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("Model trained successfully!")

    # ====================================================
    # 6️⃣ Evaluation Metrics
    # ====================================================


    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.metric("Accuracy", f"{accuracy:.2%}")

    st.text("Classification Report")
    st.json(report)

    st.text("Confusion Matrix")
    conf_df = pd.DataFrame(
        conf_matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"]
    )
    st.write(conf_df)

    # Heatmap for nicer visualization
    st.subheader("Confusion Matrix Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

    st.success("✅ Internal evaluation complete. Metrics are based on pseudo VADER labels.")
else:
    st.info("Upload a CSV file to start analysis. Make sure it contains a 'body' column.")
