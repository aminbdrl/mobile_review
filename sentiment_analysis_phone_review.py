import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ===============================
# NLTK DOWNLOADS
# ===============================
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

analyzer = SentimentIntensityAnalyzer()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="VoC Sentiment Analysis",
    page_icon="🕸",
    layout="wide"
)

st.title("Know Your Customers: Sentiment Analysis Made Simple")
st.markdown("------------------------------------------------------------")

# ===============================
# HELPER FUNCTION
# ===============================
def vader_predict(text):
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score >= 0.5 else "Negative"

# ===============================
# FILE UPLOAD
# ===============================
file = st.sidebar.file_uploader("Mobile_reviews.csv", type=["csv"])

if file is not None:
    data = pd.read_csv(file)
    data["body"] = data["body"].astype(str)

    # ===============================
    # SENTIMENT ANALYSIS
    # ===============================
    data["score"] = data["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    data["sentiment"] = data["score"].apply(lambda x: "Positive" if x >= 0.5 else "Negative")
    data["date"] = pd.to_datetime(data["date"])

    # ===============================
    # SIDEBAR STATS
    # ===============================
    st.sidebar.subheader("Review Count by Brand")
    brand_counts = data["brand"].value_counts()
    for brand, count in brand_counts.items():
        st.sidebar.write(f"{brand}: {count}")

    # ===============================
    # PIE + TREND (NOKIA)
    # ===============================
    st.subheader("Sentiment Distribution & Trend (Nokia)")
    col1, col2 = st.columns(2)

    with col1:
        nokia = data[data["brand"] == "Nokia"]
        pie = nokia["sentiment"].value_counts().reset_index()
        pie.columns = ["Sentiment", "Count"]
        fig = px.pie(pie, names="Sentiment", values="Count", title="Nokia Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        nokia["Month"] = nokia["date"].dt.strftime("%m-%Y")
        trend = nokia.groupby(["Month", "sentiment"]).size().reset_index(name="Count")
        fig2 = px.line(trend, x="Month", y="Count", color="sentiment",
                       title="Sentiment Trend Over Time (Nokia)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("------------------------------------------------------------")

    # ===============================
    # DISTRIBUTION CHARTS
    # ===============================
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(
            data, x="brand", color="sentiment",
            title="Sentiment Count by Brand"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        percent = data.groupby(["brand", "sentiment"]).size().reset_index(name="Count")
        total = data.groupby("brand").size().reset_index(name="Total")
        percent = percent.merge(total, on="brand")
        percent["Percentage"] = percent["Count"] / percent["Total"]

        fig4 = px.bar(
            percent, x="brand", y="Percentage", color="sentiment",
            title="Sentiment Percentage by Brand"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("------------------------------------------------------------")

    # ===============================
    # WORD CLOUDS
    # ===============================
    st.subheader("Word Clouds by Brand & Sentiment")

    stop_words_custom = [
        "phone", "amazon", "samsung", "nokia", "huawei", "use", "used",
        "really", "one", "still", "work", "great", "good"
    ]

    data["clean_text"] = data["body"].apply(
        lambda x: " ".join(
            [w for w in x.lower().split() if w not in stop_words_custom]
        )
    )

    for brand in ["Nokia", "HUAWEI", "Samsung"]:
        col5, col6 = st.columns(2)

        with col5:
            pos = data[(data["brand"] == brand) & (data["sentiment"] == "Positive")]
            words = " ".join(pos["clean_text"])
            wc = WordCloud(background_color="white", width=700, height=400).generate(words)
            plt.imshow(wc)
            plt.axis("off")
            plt.title(f"Positive Word Cloud ({brand})")
            st.pyplot()

        with col6:
            neg = data[(data["brand"] == brand) & (data["sentiment"] == "Negative")]
            words = " ".join(neg["clean_text"])
            wc = WordCloud(background_color="white", width=700, height=400, colormap="Reds").generate(words)
            plt.imshow(wc)
            plt.axis("off")
            plt.title(f"Negative Word Cloud ({brand})")
            st.pyplot()

    st.markdown("------------------------------------------------------------")

    # ===============================
    # TOP REVIEWS
    # ===============================
    for brand in ["Nokia", "HUAWEI", "Samsung"]:
        st.subheader(f"Top 5 Positive Reviews – {brand}")
        top_pos = data[(data["brand"] == brand) & (data["score"] > 0.9)].sort_values("score", ascending=False).head(5)
        for i, row in top_pos.iterrows():
            st.write(f"- ({row['score']:.2f}) {row['body']}")

        st.subheader(f"Top 5 Negative Reviews – {brand}")
        top_neg = data[(data["brand"] == brand) & (data["score"] < 0.1)].sort_values("score").head(5)
        for i, row in top_neg.iterrows():
            st.write(f"- ({row['score']:.2f}) {row['body']}")

    st.markdown("============================================================")
    st.subheader("Model Performance Evaluation")

    # ===============================
    # PERFORMANCE EVALUATION
    # ===============================
    eval_file = st.file_uploader(
        "Upload labeled data for evaluation (CSV with body, true_sentiment):",
        type="csv"
    )

    if eval_file is not None:
        eval_df = pd.read_csv(eval_file)
        eval_df["body"] = eval_df["body"].astype(str)
        eval_df["predicted_sentiment"] = eval_df["body"].apply(vader_predict)

        y_true = eval_df["true_sentiment"]
        y_pred = eval_df["predicted_sentiment"]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label="Positive")
        rec = recall_score(y_true, y_pred, pos_label="Positive")
        f1 = f1_score(y_true, y_pred, pos_label="Positive")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Accuracy", f"{acc:.2f}")
        colB.metric("Precision", f"{prec:.2f}")
        colC.metric("Recall", f"{rec:.2f}")
        colD.metric("F1-Score", f"{f1:.2f}")

        cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Negative"])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Positive", "Actual Negative"],
            columns=["Predicted Positive", "Predicted Negative"]
        )

        st.subheader("Confusion Matrix")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig_cm)
