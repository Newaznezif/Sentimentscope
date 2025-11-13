import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
import numpy as np
from streamlit_lottie import st_lottie
import requests

nltk.download('stopwords')

# -----------------------------
# Load model & vectorizer
# -----------------------------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stop_words = set(stopwords.words('english'))

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stop_words])

# -----------------------------
# Load Lottie animation
# -----------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_man = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_2ys3w4dj.json")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="SentimentScope", page_icon="📝", layout="wide")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* Body */
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f5f5f5; }

/* Navbar */
.navbar { display: flex; justify-content: space-between; align-items: center; padding: 15px 50px; background: #1f2937; color: #fff; font-weight: bold; position: sticky; top: 0; z-index: 999; box-shadow: 0px 4px 12px rgba(0,0,0,0.2);}
.navbar a { color: #fff; text-decoration: none; margin-left: 25px; transition: all 0.3s ease;}
.navbar a:hover { color: #60a5fa; transform: scale(1.1); }

/* Main Title */
.title { text-align: center; margin: 50px 0 20px; font-size: 3em; color: #1f2937; }

/* Subtitle */
.subtitle { text-align: center; font-size: 1.5em; color: #4b5563; margin-bottom: 40px; }

/* Centered textarea */
.stTextArea>div>div>textarea {
    border-radius: 15px !important;
    padding: 15px !important;
    font-size: 1.2em !important;
    width: 70% !important;
    max-width: 800px !important;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    color: white;
    font-size: 1.2em;
    font-weight: bold;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #60a5fa, #3b82f6);
}

/* Result card */
.result-card {
    width: 70%;
    max-width: 800px;
    margin: 20px auto;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 1.5em;
    color: #1f2937;
    box-shadow: 0px 4px 30px rgba(0,0,0,0.1);
}

/* Footer */
.footer { text-align: center; margin-top: 60px; padding: 25px; background-color: #1f2937; color: white;}
.footer a { margin: 0 10px; color: #60a5fa; text-decoration: none; transition: all 0.3s ease; }
.footer a:hover { transform: scale(1.1); }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Navbar HTML
# -----------------------------
st.markdown("""
<div class="navbar">
    <div><strong>SentimentScope 📝</strong></div>
    <div>
        <a href="#">Home</a>
        <a href="#contact-section">Contact</a>
        <a href="#sentiment-tool">Tool</a>
        <a href="https://twitter.com/" target="_blank">Twitter</a>
        <a href="https://www.linkedin.com/in/newaz-nezif-285439262/" target="_blank">LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Section
# -----------------------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<h1 class="title">SentimentScope 📝</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="subtitle">Analyze text sentiment instantly!</h4>', unsafe_allow_html=True)
if lottie_man:
    st_lottie(lottie_man, height=200)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Sentiment Tool
# -----------------------------
st.markdown('<a name="sentiment-tool"></a>', unsafe_allow_html=True)
user_input = st.text_area("Enter your text here:", height=150, placeholder="Type something amazing...", label_visibility="collapsed")

if st.button("Analyze Sentiment") and user_input.strip():
    clean_text = preprocess(user_input)
    vect = vectorizer.transform([clean_text])
    probs = model.predict_proba(vect)[0]
    prediction = model.classes_[np.argmax(probs)]
    confidence = np.max(probs) * 100
    sentiment = "neutral" if confidence < 60 else prediction

    emoji = "😃" if sentiment=="positive" else "😡" if sentiment=="negative" else "😐"
    color = "#90ee90" if sentiment=="positive" else "#ffcccb" if sentiment=="negative" else "#d3d3d3"

    st.markdown(
        f'<div class="result-card" style="background-color:{color};">{emoji} {sentiment.upper()} ({confidence:.2f}% confidence)</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div id="contact-section" class="footer">
    &copy; 2025 Newaz. All rights reserved.
    <br>Contact me:
    <a href='https://twitter.com/' target='_blank'>\U0001F426 Twitter</a> |
    <a href='https://www.linkedin.com/in/newaz-nezif-285439262//' target='_blank'>\U0001F4BC LinkedIn</a> |
    <a href='https://github.com/' target='_blank'>\U0001F419 GitHub</a>
</div>
""", unsafe_allow_html=True)

