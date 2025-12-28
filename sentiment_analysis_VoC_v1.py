import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# --------------------------
# Streamlit page config & options
# --------------------------
st.set_page_config(
    page_title="VoC sentiment analysis",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress Pyplot deprecation warning globally
st.set_option('deprecation.showPyplotGlobalUse', False)

# --------------------------
# App title
# --------------------------
st.title("VoC: Sentiment Analysis POC")
st.markdown("------------------------------------------------------------------------------------")

# --------------------------
# File upload
# --------------------------
filename = st.sidebar.file_uploader("Upload reviews data:", type=("csv", "xlsx"))

if filename is not None:
    data = pd.read_csv(filename)
    data["body"] = data["body"].astype("str")
    data["score"] = data["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    data["sentiment"] = np.where(data['score'] >= 0.5, "Positive", "Negative")
    data = data[['brand','body','sentiment','score','date']]
    data['date'] = pd.to_datetime(data['date'])
    data['quarter'] = pd.PeriodIndex(data.date, freq='Q')

    # --------------------------
    # Sentiment percentage
    # --------------------------
    per_dt = data.groupby(['brand','sentiment']).size().reset_index(name='count')
    per_dt1 = data.groupby(['brand']).size().reset_index(name='total')
    per_dt2 = pd.merge(per_dt, per_dt1, how='left', on='brand')
    per_dt2['Sentiment_Percentage'] = per_dt2['count'] / per_dt2['total']

    # --------------------------
    # Sidebar: reviews count
    # --------------------------
    brand_c = data.groupby(['brand']).size().reset_index(name='count')
    st.sidebar.write("Reviews count by brand:")
    for i, row in brand_c.iterrows():
        st.sidebar.write(f"{row['brand']} : {row['count']}")

    # --------------------------
    # Sentiment distribution plots
    # --------------------------
    st.subheader("Phone Reviews Sentiment distribution")
    col3, col4 = st.columns(2)

    with col4:
        data1 = data[data['brand'] == 'Nokia']
        sentiment_count = data1['sentiment'].value_counts().reset_index()
        sentiment_count.columns = ['Sentiments', 'Count']
        fig = px.pie(sentiment_count, values='Count', names='Sentiments',
                     width=550, height=400,
                     title='Sentiment distribution for Nokia').update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        trend_dt = data[data['brand'] == 'Nokia']
        trend_dt['Review_Month'] = trend_dt['date'].dt.strftime('%m-%Y')
        trend_dt1 = trend_dt.groupby(['Review_Month','sentiment']).size().reset_index(name='Sentiment_Count')
        trend_dt1 = trend_dt1.sort_values(['sentiment'], ascending=False)
        fig2 = px.line(trend_dt1, x="Review_Month", y="Sentiment_Count", color='sentiment',
                       width=600, height=400,
                       title='Trend analysis of sentiments for Nokia').update_layout(title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("------------------------------------------------------------------------------------")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x="brand", y="sentiment",
                           histfunc="count", color="sentiment", facet_col="sentiment",
                           labels={"sentiment": "sentiment"},
                           width=550, height=400).update_layout(title_text='Distribution by count of sentiment', title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig1 = px.histogram(per_dt2, x="brand", y="Sentiment_Percentage", color="sentiment",
                            facet_col="sentiment", labels={"sentiment": "sentiment"},
                            width=550, height=400).update_layout(yaxis_title="Percentage",
                                                                 title_text='Distribution by percentage of sentiment', title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Word Cloud for reviews Sentiment")

    word_ls = ['phone.','phone,','will','window','really','andoid','tracfone','minute','best','time','amazon','need','still','work','phone','huawei','samsung','nokia','windows phone','great','good','use','love','one','amazing','still used','lumia','iphone']
    data['body1'] = data['body'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in (word_ls)]))
    data['body1'] = data['body1'].str.replace('phone',' ', regex=True)

    # Function to generate word clouds
    def plot_wordcloud(df, title, colormap="viridis"):
        words = " ".join(df["body1"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white",
                              width=800, height=640, colormap=colormap).generate(words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        st.pyplot()

    # Display word clouds by brand and sentiment
    brands = ['Nokia', 'HUAWEI', 'Samsung']
    sentiments = [('Positive', '>0.9', "viridis"), ('Negative', '<=0.2', "RdYlGn")]

    for i in range(0, len(brands), 2):
        colA, colB = st.columns(2)
        for idx, brand in enumerate(brands[i:i+2]):
            with [colA, colB][idx]:
                for sentiment, score_cond, cmap in sentiments:
                    if sentiment == 'Positive':
                        df = data[(data['brand']==brand) & (data['sentiment']==sentiment) & (data['score']>0.9)]
                    else:
                        df = data[(data['brand']==brand) & (data['sentiment']==sentiment) & (data['score']<=0.2)]
                    plot_wordcloud(df, f"{sentiment} reviews word cloud for {brand}", colormap=cmap)

    st.markdown("------------------------------------------------------------------------------------")

    # --------------------------
    # Top 5 positive & negative reviews for Nokia
    # --------------------------
    st.subheader("Top 5 positive reviews for Nokia :")
    pos = data[(data['brand'] == 'Nokia') & (data['score'] > 0.9)].sort_values('score', ascending=False).head(5).reset_index()
    for i, row in pos.iterrows():
        st.write(f"{i+1}. {row['brand']} | Positive | Sentiment Score: {row['score']} - {row['body']}")

    st.markdown("------------------------------------------------------------------------------------")
    st.subheader("Top 5 negative reviews for Nokia :")
    neg = data[(data['brand'] == 'Nokia') & (data['score'] < 0.1)].sort_values('score').head(5).reset_index()
    for i, row in neg.iterrows():
        st.write(f"{i+1}. {row['brand']} | Negative | Sentiment Score: {row['score']} - {row['body']}")
