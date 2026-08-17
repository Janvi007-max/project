"""
Simple Sentiment Classifier — Web UI Version (Streamlit)
--------------------------------------------------------------
Same model and logic as simple_sentiment.py, just with a web page
on top instead of typing in the terminal.

Why Streamlit? It turns a plain Python script into a web app with
almost no extra code — no HTML, no CSS, no JavaScript needed.

Install once (if needed):
    pip install scikit-learn streamlit

Run:
    streamlit run simple_sentiment_ui.py

This opens a browser tab automatically at http://localhost:8501
"""

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------------------------------
# STEP 1: Our training data (same small demo dataset)
# -----------------------------------------------------
sentences = [
    "I love this movie, it was amazing",
    "What a great day, I feel wonderful",
    "This is the best food I have ever had",
    "I am so happy with the service",
    "Excellent product, highly recommend",
    "The staff were friendly and helpful",
    "I really enjoyed the concert last night",
    "This app is fantastic and easy to use",

    "I hate this movie, it was terrible",
    "What a bad day, I feel awful",
    "This is the worst food I have ever had",
    "I am so unhappy with the service",
    "Terrible product, do not recommend",
    "The staff were rude and unhelpful",
    "I really disliked the concert last night",
    "This app is horrible and confusing",
]
labels = [1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0]


# -----------------------------------------------------
# STEP 2: Train the model once, then reuse it
# -----------------------------------------------------
# @st.cache_resource means Streamlit trains the model only once
# and reuses it, instead of retraining every time you type something.

@st.cache_resource
def train_model():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    y = labels

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return vectorizer, model, accuracy


vectorizer, model, accuracy = train_model()


def predict_sentiment(text):
    text_as_numbers = vectorizer.transform([text])
    result = model.predict(text_as_numbers)[0]
    probabilities = model.predict_proba(text_as_numbers)[0]
    confidence = max(probabilities)
    label = "Positive 😊" if result == 1 else "Negative 😞"
    return label, confidence


# -----------------------------------------------------
# STEP 3: The web page itself
# -----------------------------------------------------
st.set_page_config(page_title="Simple Sentiment Classifier", page_icon="💬")

st.title("💬 Simple Sentiment Classifier")
st.write("A beginner NLP project — trained on a small example dataset "
         "using **CountVectorizer + Naive Bayes**.")

st.info(f"Model accuracy on test data: **{accuracy * 100:.0f}%** "
        f"(small demo dataset, so this is just for learning)")

st.divider()

user_text = st.text_input("Type a sentence to check its sentiment:")

if user_text:
    label, confidence = predict_sentiment(user_text)
    st.subheader(f"Prediction: {label}")
    st.progress(confidence)
    st.caption(f"Confidence: {confidence * 100:.0f}%")

st.divider()
with st.expander("How does this work?"):
    st.markdown("""
    1. **CountVectorizer** turns each sentence into word-count numbers.
    2. **Naive Bayes** learns which words appear more often in positive
       vs negative sentences.
    3. When you type a new sentence, it's converted the same way and
       the model predicts based on what it learned.
    """)
