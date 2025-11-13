# predict_sentiment.py
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("Enter 'exit' to quit the program.")
while True:
    text = input("Enter a sentence: ")
    if text.lower() == 'exit':
        break
    clean_text = preprocess(text)
    vect_text = vectorizer.transform([clean_text])
    prediction = model.predict(vect_text)[0]
    print(f"Sentiment: {prediction}\n")
