# sentiment_analyzer.py
# Simple, clean version for IMDB 50K dataset only

# Step 1: Imports
import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 2: Download NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 3: Load IMDB dataset
# Make sure "IMDB Dataset.csv" is in the same folder
df = pd.read_csv("IMDB Dataset.csv")
print("Columns in IMDB Dataset:", df.columns.tolist())

# Keep only the relevant columns and rename them
df = df[['review', 'sentiment']]
df = df.rename(columns={'review': 'text', 'sentiment': 'label'})

# Step 4: Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

print("Preprocessing text... (this may take a bit)")
df['clean_text'] = df['text'].apply(preprocess)

# Step 5: Features and labels
X = df['clean_text']
y = df['label']

# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Step 8: Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 9: Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\n✅ Model and vectorizer saved successfully!")
