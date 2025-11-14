import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk
import numpy as np
import requests
from datetime import datetime
import os

# Download stopwords with error handling
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Error loading stopwords: {e}")
    # Fallback stopwords
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
        'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
        'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
        'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
    }

# -----------------------------
# Cache model loading with error handling
# -----------------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()

# Check if models loaded successfully
if model is None or vectorizer is None:
    st.error("❌ Failed to load ML models. Please check if sentiment_model.pkl and vectorizer.pkl exist.")
    st.stop()

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
        if confidence > 80: return "Very Positive"
        elif confidence > 60: return "Positive"
        else: return "Slightly Positive"
    else:
        if confidence > 80: return "Very Negative"
        elif confidence > 60: return "Negative"
        else: return "Slightly Negative"

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

lottie_analyze = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_2ys3w4dj.json")

# -----------------------------
# Initialize session state
# -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_filter' not in st.session_state:
    st.session_state.current_filter = "all"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Sentiment", 
    page_icon="🧠", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)


# -----------------------------
# Load External CSS
# -----------------------------
def load_css():
    css_file = os.path.join("templates", "styles.css")
    if os.path.exists(css_file):
        try:
            with open(css_file, "r", encoding='utf-8') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading CSS: {e}")
            load_fallback_css()
    else:
        load_fallback_css()

def load_fallback_css():
    
    
    
    
    
    st.markdown("""
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
        }
        .main .block-container { padding-top: 0; }
        #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

load_css()

# -----------------------------
# HTML Structure in Streamlit
# -----------------------------

# Navbar
st.markdown("""
<div class="navbar">
    <div class="nav-container">
        <div class="nav-brand">🧠 Sentiments</div>
        <div class="nav-links">
            <a href="#analysis-section" class="nav-link">Analyze</a>
            <a href="#history-section" class="nav-link">History</a>
            <a href="#about-section" class="nav-link">About</a>
            <a href="#contact-section" class="nav-link">Contact</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<style>
.hero-section { padding: 20px; background-color:#f7fafc; border-radius:100px; }
.logo-section { text-align:center; margin-bottom:20px; }
.logo { font-size:32px; color:#2d3748; }
.hero-subtitle { font-size:18px; color:#4a5568; }
.stats-container { display:flex; justify-content:center; gap:50px; font-weight:bold; }
.stat-item { text-align:center; }
.stat-number { font-size:24px; }
</style>

<div class="hero-section">
    <div class="logo-section">
        <div class="logo">Sentiments</div>
        <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number" style="color: #48bb78;">98%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" style="color: #667eea;">10K+</div>
            <div class="stat-label">Analyses</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" style="color: #ed8936;">0.2s</div>
            <div class="stat-label">Response Time</div>
        </div>
    </div>
        <div class="hero-subtitle">AI-Powered Text Emotion Analysis</div>
    
</div>
""", unsafe_allow_html=True)

# Lottie Animation
if lottie_analyze:
    try:
        st_lottie(lottie_analyze, height=300, key="hero_lottie")
    except Exception as e:
        st.warning("Lottie animation couldn't be loaded")

# -----------------------------
# Analysis Section
# -----------------------------
st.markdown("""<a name="analysis-section"></a>""", unsafe_allow_html=True)

# File upload option
uploaded_file = st.file_uploader("📁 Upload a text file", type=['txt'], help="Upload .txt files for analysis")

user_input = ""
if uploaded_file is not None:
    try:
        text = str(uploaded_file.read(), "utf-8")
        user_input = st.text_area("**File Content:**", value=text, height=200, placeholder="Your text will appear here...")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    user_input = st.text_area("**Type your text here:**", height=200, placeholder="Share your thoughts... I'm here to understand the emotions behind your words! 😊")

# Analysis buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_clicked = st.button("🚀 Analyze Sentiment", use_container_width=True)

# Real-time analysis preview
if user_input and len(user_input.split()) > 3:
    with st.expander("🔍 Real-time Analysis Preview", expanded=True):
        clean_text = preprocess(user_input)
        if len(clean_text.split()) > 1:
            try:
                vect = vectorizer.transform([clean_text])
                probs = model.predict_proba(vect)[0]
                prediction = model.classes_[np.argmax(probs)]
                confidence = np.max(probs) * 100
                
                st.progress(int(confidence))
                st.caption(f"**Current detection:** {prediction.title()} ({confidence:.1f}% confidence)")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    # Safe probability calculation
                    positive_idx = np.where(model.classes_ == "positive")[0]
                    positive_prob = probs[positive_idx[0]] * 100 if len(positive_idx) > 0 else 0
                    st.metric("Positive", f"{positive_prob:.1f}%")
                with col_b:
                    negative_idx = np.where(model.classes_ == "negative")[0]
                    negative_prob = probs[negative_idx[0]] * 100 if len(negative_idx) > 0 else 0
                    st.metric("Negative", f"{negative_prob:.1f}%")
                    
            except Exception as e:
                st.error(f"Preview analysis error: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# Sentiment Analysis Results
if analyze_clicked and user_input.strip():
    try:
        if len(user_input.strip().split()) < 2:
            st.warning("⚠️ Please enter more text for accurate analysis! (At least 2 words)")
        else:
            with st.spinner('🔮 Analyzing the emotions in your text...'):
                clean_text = preprocess(user_input)
                vect = vectorizer.transform([clean_text])
                probs = model.predict_proba(vect)[0]
                prediction = model.classes_[np.argmax(probs)]
                confidence = np.max(probs) * 100
                
                detailed_sentiment = get_detailed_sentiment(prediction, confidence)
                
                sentiment_class = "positive-result" if "positive" in detailed_sentiment.lower() else "negative-result"
                emoji = "😊" if "positive" in detailed_sentiment.lower() else "😞"
                if "very" in detailed_sentiment.lower():
                    emoji = "😍" if "positive" in detailed_sentiment.lower() else "😠"
                elif "slightly" in detailed_sentiment.lower():
                    emoji = "🙂" if "positive" in detailed_sentiment.lower() else "😕"
                
                st.markdown(f"""
                <div class="result-card {sentiment_class}">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">{emoji}</div>
                    <h2 style="margin: 0; font-size: 2.5rem;">{detailed_sentiment.upper()}</h2>
                    <div style="font-size: 1.2rem; opacity: 0.8; margin-top: 0.5rem;">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Save to history
                st.session_state.history.append({
                    'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'sentiment': detailed_sentiment,
                    'confidence': f"{confidence:.1f}%",
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'full_sentiment': prediction
                })
                
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")

# -----------------------------
# About Section
# -----------------------------
st.markdown("""<a name="about-section"></a>""", unsafe_allow_html=True)
st.markdown("""

""", unsafe_allow_html=True)



st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# History Section
# -----------------------------
st.markdown("""<a name="history-section"></a>""", unsafe_allow_html=True)
# Filter buttons
st.subheader("Filter by Sentiment:")
filter_cols = st.columns(3)
with filter_cols[0]:
    if st.button("All", use_container_width=True):
        st.session_state.current_filter = "all"
        st.rerun()
with filter_cols[1]:
    if st.button("Positive", use_container_width=True):
        st.session_state.current_filter = "positive"
        st.rerun()
with filter_cols[2]:
    if st.button("Negative", use_container_width=True):
        st.session_state.current_filter = "negative"
        st.rerun()

st.info(f"Showing: {st.session_state.current_filter.title()} analyses")
if st.session_state.history:
    if st.button("🗑️ Clear All History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    
    filtered_history = st.session_state.history
    if st.session_state.current_filter != "all":
        filtered_history = [h for h in st.session_state.history if st.session_state.current_filter in h['full_sentiment'].lower()]
    
    for i, entry in enumerate(reversed(filtered_history[-10:])):
        history_class = "history-positive" if "positive" in entry['full_sentiment'].lower() else "history-negative"
        
        with st.expander(f"{entry['timestamp']} - {entry['sentiment']} ({entry['confidence']})", expanded=i==0):
            st.markdown(f"""
            <div class="history-item {history_class}">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <strong>Text:</strong> {entry['text']}
                    </div>
                    <div style="text-align: right; margin-left: 1rem;">
                        <div style="font-weight: 600; color: {'#48bb78' if 'positive' in entry['full_sentiment'].lower() else '#e53e3e'}">
                            {entry['sentiment']}
                        </div>
                        <div style="color: #718096; font-size: 0.9rem;">
                            {entry['confidence']} confidence
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("📝 No analysis history yet. Analyze some text to see your history here!")

st.markdown("</div>", unsafe_allow_html=True)
# -----------------------------
# Footer
# -----------------------------
st.markdown("""<a name="contact-section"></a>""", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <h3>🔗 Connect With Me</h3>
    <p>Made with ❤️ by Newaz | Advanced Sentiment Analysis Powered by Machine Learning</p>
    <div class="social-links">
        <a href='https://x.com/newaznezif53' target='_blank'>🐦 Twitter</a>
        <a href='https://www.linkedin.com/in/newaz-nezif-285439262/' target='_blank'>💼 LinkedIn</a>
        <a href='https://github.com/Newaznezif' target='_blank'>🐙 GitHub</a>
        <a href='newaznezif@gmail.com'>📧 Email</a>
    </div>
    <small>&copy; 2025 Sentiments. All rights reserved.</small>
</div>
""", unsafe_allow_html=True)