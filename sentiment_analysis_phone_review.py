# ====================================================
# 📌 Mobile Brand Sentiment Analyzer (Nokia, Huawei, Samsung)
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
    page_title="Mobile Brand Sentiment Analyzer",
    layout="wide"
)
st.title("📱 Brand-Specific Mobile Reviews Sentiment Analysis")
st.markdown("""
This app generates **pseudo-labels** using VADER, trains a **Naive Bayes model**,
and evaluates **sentiment separately for Nokia, Huawei, and Samsung** reviews.
""")

# ====================================================
# 2️⃣ Load Dataset
# ====================================================
uploaded_file = st.file_uploader("Upload CSV with columns 'body' and 'brand'", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna(subset=["body", "brand"])
    st.success(f"Dataset loaded: {df.shape[0]} rows")
    st.write(df.head())

    # ====================================================
    # 3️⃣ Text Cleaning
    # ====================================================
    def clean_text(text):
        text = str(text).lower()
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
            return "Neutral"

    df["true_sentiment"] = df["clean_body"].apply(vader_label)

    # ====================================================
    # 5️⃣ Function to Train and Evaluate Model per Brand
    # ====================================================
    def analyze_brand(brand_name):
        st.header(f"📊 Analysis for {brand_name}")
        brand_df = df[df["brand"].str.lower() == brand_name.lower()]
        
        if brand_df.empty:
            st.warning(f"No reviews found for {brand_name}")
            return

        # Display label distribution
        st.write("Label distribution:")
        st.bar_chart(brand_df["true_sentiment"].value_counts())

        # Prepare features
        X = brand_df["clean_body"]
        y_true = brand_df["true_sentiment"]

        vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_true, test_size=0.2, random_state=42
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=["Negative", "Neutral", "Positive"])
        report = classification_report(y_test, y_pred, output_dict=True)

        st.metric("Accuracy", f"{accuracy:.2%}")
        st.text("Classification Report")
        st.json(report)

        st.text("Confusion Matrix")
        conf_df = pd.DataFrame(
            conf_matrix,
            index=["Actual Negative", "Actual Neutral", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Neutral", "Predicted Positive"]
        )
        st.write(conf_df)

        # Heatmap
        st.subheader("Confusion Matrix Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        st.pyplot(fig)

    # ====================================================
    # 6️⃣ Run Analysis for Each Brand
    # ====================================================
    for brand in ["Nokia", "Huawei", "Samsung"]:
        analyze_brand(brand)

else:
    st.info("Upload a CSV file to start analysis. Make sure it contains 'body' and 'brand' columns.")
