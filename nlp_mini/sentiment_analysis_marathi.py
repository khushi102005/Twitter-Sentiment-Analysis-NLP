# sentiment_analysis_marathi.py
import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Transformers for LLM model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Force slow tokenizer
def load_llm_pipeline():
    # Using a multilingual BERT model (works for Marathi)
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

# -----------------------------
# Marathi stopwords (custom)
# -----------------------------
MARATHI_STOPWORDS = set([
'आहे', 'आहेत', 'असले', 'असते', 'तू', 'तो', 'ती', 
    'ते', 'आणि', 'पण', 'साठी', 'किवा', 'म्हणून', 'करून', 
    'तेव्हा', 'कधी', 'जसे', 'तसे'
])

# -----------------------------
# Text cleaning function
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\u0900-\u097F\s]", "", text)  # remove URLs, mentions, hashtags, special chars
    tokens = text.split()  # simple whitespace tokenizer
    tokens = [t for t in tokens if t not in MARATHI_STOPWORDS]
    return " ".join(tokens)

# -----------------------------
# Load sample dataset
# -----------------------------
@st.cache_data
def load_dataset(csv_path="data/tweets-train.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['tweet','label']).reset_index(drop=True)
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    return df


# -----------------------------
# Train NLTK (Naive Bayes) model
# -----------------------------
@st.cache_resource
def train_nltk_model(df):
    X = df['clean_tweet']
    y = df['label']  # use numeric labels directly

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline_model = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,2))),  # unigrams + bigrams
        ('clf', MultinomialNB())
    ])
    pipeline_model.fit(X_train, y_train)
    return pipeline_model  # <-- return the whole pipeline, not just vectorizer



# -----------------------------
# Load LLM sentiment pipeline
# -----------------------------
@st.cache_resource
def load_llm_pipeline():
    # Using a multilingual model
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# -----------------------------
# Streamlit GUI
# -----------------------------
st.title("📝 Marathi Twitter Sentiment Analysis")
st.write("Choose a model and input a tweet to get sentiment prediction.")

# Model selection
model_choice = st.radio("Select Model", ["NLTK Naive Bayes", "LLM Transformer"])

# Load dataset and train NLTK model
if model_choice == "NLTK Naive Bayes":
    df = load_dataset()
    nltk_model = train_nltk_model(df) 

# Load LLM model
if model_choice == "LLM Transformer":
    llm_model = load_llm_pipeline()

# Input tweet
tweet_input = st.text_area("Enter Marathi tweet here", value="आजचा दिवस छान आहे!")

if st.button("Predict Sentiment"):
    if not tweet_input.strip():
        st.warning("Please enter a tweet!")
    else:
        if model_choice == "NLTK Naive Bayes":
            clean_tweet = clean_text(tweet_input)
            pred_label = nltk_model.predict([clean_tweet])[0]  # works now
            label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
            st.success(f"✅ Sentiment: {label_map.get(pred_label, pred_label)}")

        elif model_choice == "LLM Transformer":
            result = llm_model(tweet_input)
            star_label = result[0]['label']  # e.g., '5 stars'
            score = result[0]['score']

            # Map 1-5 stars to numeric labels
            def map_star_to_label(star_label):
                star_num = int(star_label[0])  # '5 stars' -> 5
                if star_num <= 2:
                    return -1  # Negative
                elif star_num == 3:
                    return 0   # Neutral
                else:
                    return 1   # Positive

            pred_label = map_star_to_label(star_label)
            label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
            st.success(f"✅ Sentiment: {label_map.get(pred_label)}, confidence: {score:.2f}")