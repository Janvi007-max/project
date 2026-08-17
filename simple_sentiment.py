"""
Simple Sentiment Classifier — Beginner-Friendly NLP Project
--------------------------------------------------------------
What this project does, in plain words:
  1. We have a list of sentences, each labeled "positive" or "negative".
  2. We turn each sentence into numbers (computers can't read words directly).
  3. We train a simple ML model to learn which words mean "positive"
     and which mean "negative".
  4. We test the model on new sentences it has never seen.
  5. We let YOU type a sentence and the model predicts the sentiment.

No deep learning, no TensorFlow — just scikit-learn, which is much
easier to read and explain. Every step is commented.

Install once (if needed):
    pip install scikit-learn

Run:
    python simple_sentiment.py
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------------------------------
# STEP 1: Our training data
# -----------------------------------------------------
# In a real project this would come from a CSV file.
# Here we keep it simple: a small list of example sentences.

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

# 1 = positive, 0 = negative (same order as the sentences above)
labels = [1, 1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0, 0]


# -----------------------------------------------------
# STEP 2: Turn words into numbers
# -----------------------------------------------------
# CountVectorizer looks at every sentence and counts how many times
# each word appears. This turns text into a table of numbers the
# model can actually learn from.

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)   # X = word counts for each sentence
y = labels                                # y = the correct answers


# -----------------------------------------------------
# STEP 3: Split into training and testing data
# -----------------------------------------------------
# We train on most of the data, and keep a small portion aside to
# check if the model actually learned something, instead of just
# memorizing.

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# -----------------------------------------------------
# STEP 4: Train the model
# -----------------------------------------------------
# Naive Bayes is a simple, classic algorithm that works surprisingly
# well for text. It looks at word frequencies and calculates the
# probability that a sentence is positive or negative.

model = MultinomialNB()
model.fit(x_train, y_train)


# -----------------------------------------------------
# STEP 5: Check how well it learned
# -----------------------------------------------------
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy on test data: {accuracy * 100:.0f}%")


# -----------------------------------------------------
# STEP 6: Try it yourself (the "interface")
# -----------------------------------------------------
def predict_sentiment(text):
    """Takes a sentence and returns 'Positive' or 'Negative'."""
    text_as_numbers = vectorizer.transform([text])   # same step as STEP 2
    result = model.predict(text_as_numbers)[0]        # 1 or 0
    return "Positive" if result == 1 else "Negative"


print("\nType a sentence to check its sentiment.")
print("Type 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit", ""):
        print("Goodbye!")
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted sentiment: {sentiment}\n")
