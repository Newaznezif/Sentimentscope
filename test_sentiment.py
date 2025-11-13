import joblib
import string
from nltk.corpus import stopwords
import numpy as np

# Load saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load stopwords
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

while True:
    sentence = input("Enter a sentence to analyze sentiment (or type 'exit' to quit): ")
    if sentence.lower() == 'exit':
        print("Exiting...")
        break
    if sentence.strip() == '':
        print("You need to type a sentence!")
        continue

    clean_sentence = preprocess(sentence)
    vect_sentence = vectorizer.transform([clean_sentence])

    # Get probabilities and prediction
    probs = model.predict_proba(vect_sentence)[0]
    prediction = model.classes_[np.argmax(probs)]
    confidence = np.max(probs) * 100

    # Neutral fallback for low-confidence predictions
    if confidence < 60:
        sentiment = "neutral"
    else:
        sentiment = prediction

    print(f"Sentiment: {sentiment} ({confidence:.2f}% confidence)\n")
