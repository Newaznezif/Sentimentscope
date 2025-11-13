import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
import numpy as np
from streamlit_lottie import st_lottie
import requests
from datetime import datetime

# Download stopwords
nltk.download('stopwords')

# -----------------------------
# Cache model loading
# -----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stop_words])

# -----------------------------
# Enhanced sentiment mapping
# -----------------------------
def get_detailed_sentiment(prediction, confidence):
    if prediction == "positive":
        if confidence > 80: return "Very Positive 😍"
        elif confidence > 60: return "Positive 😊"
        else: return "Slightly Positive 🙂"
    elif prediction == "negative":
        if confidence > 80: return "Very Negative 😠"
        elif confidence > 60: return "Negative 😞"
        else: return "Slightly Negative 😕"
    else:
        return "Neutral 😐"

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
# Initialize session state
# -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 60

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* Body */
body { 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    background-color: #f5f5f5; 
}

/* Navbar */
.navbar { 
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    padding: 15px 50px; 
    background: #1f2937; 
    color: #fff; 
    font-weight: bold; 
    position: sticky; 
    top: 0; 
    z-index: 999; 
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}
.navbar a { 
    color: #fff; 
    text-decoration: none; 
    margin-left: 25px; 
    transition: all 0.3s ease;
}
.navbar a:hover { 
    color: #60a5fa; 
    transform: scale(1.1); 
}

/* Main Title */
.title { 
    text-align: center; 
    margin: 30px 0 15px; 
    font-size: 3em; 
    color: #1f2937; 
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle { 
    text-align: center; 
    font-size: 1.3em; 
    color: #4b5563; 
    margin-bottom: 30px; 
}

/* Centered textarea */
.stTextArea>div>div>textarea {
    border-radius: 15px !important;
    padding: 15px !important;
    font-size: 1.1em !important;
    width: 70% !important;
    max-width: 800px !important;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    border: 2px solid #e5e7eb !important;
    transition: all 0.3s ease;
}
.stTextArea>div>div>textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0px 4px 20px rgba(59, 130, 246, 0.2) !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    color: white;
    font-size: 1.1em;
    font-weight: bold;
    padding: 12px 30px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    margin: 10px 5px;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #60a5fa, #3b82f6);
    box-shadow: 0px 6px 20px rgba(59, 130, 246, 0.4);
}

/* Result card */
.result-card {
    width: 70%;
    max-width: 800px;
    margin: 25px auto;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 1.4em;
    font-weight: bold;
    color: #1f2937;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.15);
    border: 2px solid;
    transition: all 0.3s ease;
}

/* Footer */
.footer { 
    text-align: center; 
    margin-top: 60px; 
    padding: 25px; 
    background-color: #1f2937; 
    color: white;
    border-radius: 10px 10px 0 0;
}
.footer a { 
    margin: 0 10px; 
    color: #60a5fa; 
    text-decoration: none; 
    transition: all 0.3s ease; 
    font-weight: bold;
}
.footer a:hover { 
    transform: scale(1.1); 
    color: #93c5fd;
}

/* Metrics cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    border-left: 4px solid #3b82f6;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
}
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
        <a href="#sentiment-tool">Analyze</a>
        <a href="#history-section">History</a>
        <a href="#contact-section">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# # -----------------------------
# # Hero Section
# # -----------------------------
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown('<h1 class="title">SentimentScope Analyzer</h1>', unsafe_allow_html=True)
#     st.markdown('<h4 class="subtitle">Unlock the emotions behind your text with AI-powered sentiment analysis</h4>', unsafe_allow_html=True)
    
#     # Quick stats
#     col1a, col1b, col1c = st.columns(3)
#     with col1a:
#         st.markdown('<div class="metric-card">🚀<br>Instant Analysis</div>', unsafe_allow_html=True)
#     with col1b:
#         st.markdown('<div class="metric-card">🎯<br>High Accuracy</div>', unsafe_allow_html=True)
#     with col1c:
#         st.markdown('<div class="metric-card">📊<br>Detailed Insights</div>', unsafe_allow_html=True)

# with col2:
#     if lottie_man:
#         st_lottie(lottie_man, height=250)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.session_state.confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=50, 
        max_value=90, 
        value=60,
        help="Adjust sensitivity for neutral classification"
    )
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    **SentimentScope** uses machine learning to analyze:
    - ✅ Positive sentiment
    - ❌ Negative sentiment  
    - 🔄 Neutral sentiment
    
    *Trained on thousands of text samples*
    """)
    
    st.markdown("---")
    st.header("💡 Tips")
    st.success("""
    • Longer texts = Better accuracy
    • Avoid excessive slang
    • Check confidence scores
    • Review analysis history
    """)

# -----------------------------
# Main Analysis Section
# -----------------------------
st.markdown('<a name="sentiment-tool"></a>', unsafe_allow_html=True)
st.markdown("## 🔍 Analyze Text Sentiment")

# File upload option
uploaded_file = st.file_uploader("📁 Upload a text file", type=['txt'], help="Upload .txt files for analysis")

user_input = ""
if uploaded_file is not None:
    text = str(uploaded_file.read(), "utf-8")
    user_input = st.text_area("**File Content:**", value=text, height=200, placeholder="Your text will appear here...")
else:
    user_input = st.text_area("**Or type your text here:**", height=400, width= 3000, placeholder="Type something amazing... I'm excited to analyze it! 😊")

# Analysis buttons
col_analyze, col_clear = st.columns([1, 1])
with col_analyze:
    analyze_clicked = st.button("🚀 Analyze Sentiment", use_container_width=True)
with col_clear:
    if st.button("🗑️ Clear Text", use_container_width=True):
        user_input = ""
        st.rerun()

# Real-time analysis for longer texts
if user_input and len(user_input.split()) > 5:
    with st.expander("🔍 Real-time Preview", expanded=False):
        clean_text = preprocess(user_input)
        if len(clean_text.split()) > 2:
            vect = vectorizer.transform([clean_text])
            probs = model.predict_proba(vect)[0]
            prediction = model.classes_[np.argmax(probs)]
            confidence = np.max(probs) * 100
            
            st.progress(int(confidence))
            st.caption(f"**Live detection:** {prediction.title()} ({confidence:.1f}% confidence)")

# Main analysis
if analyze_clicked and user_input.strip():
    try:
        if len(user_input.strip().split()) < 2:
            st.warning("⚠️ Please enter more text for accurate analysis! (At least 2 words)")
        else:
            with st.spinner('🔮 Analyzing sentiment...'):
                clean_text = preprocess(user_input)
                vect = vectorizer.transform([clean_text])
                probs = model.predict_proba(vect)[0]
                prediction = model.classes_[np.argmax(probs)]
                confidence = np.max(probs) * 100
                
                # Apply confidence threshold
                sentiment = "neutral" if confidence < st.session_state.confidence_threshold else prediction
                detailed_sentiment = get_detailed_sentiment(sentiment, confidence)
                
                # Determine colors and emojis
                if "positive" in detailed_sentiment.lower():
                    color = "#d1fae5"
                    border_color = "#10b981"
                    emoji = "😊"
                elif "negative" in detailed_sentiment.lower():
                    color = "#fee2e2" 
                    border_color = "#ef4444"
                    emoji = "😞"
                else:
                    color = "#f3f4f6"
                    border_color = "#6b7280"
                    emoji = "😐"
                
                # Display main result
                st.markdown(
                    f'<div class="result-card" style="background-color:{color}; border-color: {border_color};">'
                    f'{emoji} {detailed_sentiment.upper()} <br>'
                    f'<small style="font-size: 0.8em; color: #6b7280;">({confidence:.2f}% confidence)</small>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Analytics section
                st.markdown("## 📊 Detailed Analysis")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", len(user_input.split()))
                with col2:
                    st.metric("Character Count", len(user_input))
                with col3:
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                with col4:
                    reading_level = "Easy" if len(user_input.split()) < 15 else "Medium" if len(user_input.split()) < 30 else "Complex"
                    st.metric("Reading Level", reading_level)
                
                # Sentiment probabilities
                st.markdown("### Sentiment Distribution")
                prob_data = {label: prob*100 for label, prob in zip(model.classes_, probs)}
                st.bar_chart(prob_data)
                
                # Save to history
                st.session_state.history.append({
                    'text': user_input[:80] + "..." if len(user_input) > 80 else user_input,
                    'sentiment': detailed_sentiment,
                    'confidence': f"{confidence:.1f}%",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")

# -----------------------------
# History Section
# -----------------------------
st.markdown('<a name="history-section"></a>', unsafe_allow_html=True)
st.markdown("## 📜 Analysis History")

if st.session_state.history:
    # Clear history button
    if st.button("Clear All History"):
        st.session_state.history = []
        st.rerun()
    
    # Display history
    for i, entry in enumerate(reversed(st.session_state.history[-10:])):  # Last 10 entries
        with st.expander(f"{entry['timestamp']} - {entry['sentiment']} ({entry['confidence']})", expanded=i==0):
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.write(f"**Text:** {entry['text']}")
            with col_right:
                sentiment_color = {
                    'positive': '🟢',
                    'negative': '🔴', 
                    'neutral': '⚫'
                }
                # Determine color based on sentiment
                color_icon = '🟢' if 'positive' in entry['sentiment'].lower() else '🔴' if 'negative' in entry['sentiment'].lower() else '⚫'
                st.write(f"**Sentiment:** {color_icon} {entry['sentiment']}")
                st.write(f"**Confidence:** {entry['confidence']}")
else:
    st.info("📝 No analysis history yet. Analyze some text to see your history here!")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div id="contact-section" class="footer">
    <h3>🔗 Connect With Me</h3>
    <p>Made with ❤️ by Newaz | Sentiment Analysis Powered by Machine Learning</p>
    <div style="margin: 20px 0;">
        <a href='https://twitter.com/' target='_blank'>🐦 Twitter</a> |
        <a href='https://www.linkedin.com/in/newaz-nezif-285439262/' target='_blank'>💼 LinkedIn</a> |
        <a href='https://github.com/' target='_blank'>🐙 GitHub</a> |
        <a href='mailto:your-email@example.com'>📧 Email</a>
    </div>
    <small>&copy; 2025 SentimentScope. All rights reserved.</small>
</div>
""", unsafe_allow_html=True)