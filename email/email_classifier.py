import streamlit as st
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests
import json
import io

# Page configuration
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning UI
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .sample-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 4px solid #667eea;
    }
    .sample-box:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sample-btn {
        width: 100%;
        text-align: left;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for model and sample text
if 'model' not in st.session_state:
    st.session_state.model = None
if 'cv' not in st.session_state:
    st.session_state.cv = None
if 'sample_text' not in st.session_state:
    st.session_state.sample_text = ""
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Sample messages
SAMPLE_MESSAGES = {
    "🎉 Congratulations! You've won!": "Win a free iPhone now! Click here to claim your prize! Limited time offer! Urgent action required!",
    "💼 Important Meeting Update": "Hi team, the meeting scheduled for tomorrow has been moved to 3 PM in conference room B. Please update your calendars.",
    "🔒 Security Alert": "Your account has suspicious activity. Verify your identity immediately at: http://fake-security-site.com/verify-now",
    "📦 Package Delivery": "Your Amazon package #ORD-7842 will arrive today between 2-4 PM. Track your delivery here: https://amazon.com/track/ORD7842",
    "💰 Urgent Money Transfer": "You have inherited $2,500,000 from a relative. Contact lawyer immediately at lawyer@inheritance.com for transfer.",
    "📧 Normal Work Email": "Hi John, could you please send me the quarterly report when you get a chance? Thanks! Best, Sarah"
}

# Load default model function
@st.cache_resource
def load_default_model():
    try:
        model = pickle.load(open("spam.pkl", 'rb'))
        cv = pickle.load(open("vectorizer.pkl", 'rb'))
        return model, cv, True
    except FileNotFoundError:
        return None, None, False
    except Exception as e:
        st.error(f"❌ Error loading default model: {e}")
        return None, None, False

# Load custom model from uploaded files
def load_custom_model(uploaded_model_file, uploaded_vectorizer_file):
    try:
        model = pickle.load(uploaded_model_file)
        cv = pickle.load(uploaded_vectorizer_file)
        
        # Test the model with a simple prediction to verify it works
        test_text = ["hello world"]
        test_vect = cv.transform(test_text).toarray()
        test_pred = model.predict(test_vect)
        
        return model, cv, True
    except Exception as e:
        st.error(f"❌ Error loading custom model: {e}")
        return None, None, False

# Classification function
def classify_text(text, model, cv):
    try:
        data = [text]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
        probability = model.predict_proba(vect)[0]
        return pred[0], probability
    except Exception as e:
        st.error(f"❌ Classification error: {e}")
        return None, None

# Main app
def main():
    # Header section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">🛡️ SpamGuard AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Advanced Machine Learning Email Protection</p>', unsafe_allow_html=True)
    
    # Load default model if not already loaded
    if st.session_state.model is None:
        with st.spinner('🔄 Loading default model...'):
            model, cv, loaded = load_default_model()
            if loaded:
                st.session_state.model = model
                st.session_state.cv = cv
                st.session_state.model_loaded = True
                st.success("✅ Default model loaded successfully!")
            else:
                st.info("📝 Please upload your model files to get started")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter Email Text")
        
        # Text area with sample text from session state
        user_input = st.text_area(
            "",
            height=200,
            value=st.session_state.sample_text,
            placeholder="Paste your email content here...",
            help="Enter the email text you want to check for spam",
            key="text_input"
        )
        
        # Classification button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            classify_btn = st.button("🚀 Analyze Email", use_container_width=True)
        
        # Results section
        if classify_btn and user_input:
            if st.session_state.model_loaded and st.session_state.model and st.session_state.cv:
                with st.spinner('🔍 Analyzing content...'):
                    pred, probability = classify_text(user_input, st.session_state.model, st.session_state.cv)
                    
                    if pred is not None:
                        # Display results with beautiful styling
                        if pred == 0:
                            st.markdown(f"""
                            <div class="result-box" style="background: linear-gradient(45deg, #56ab2f, #a8e063); color: white;">
                                ✅ SAFE EMAIL<br>
                                <small>Confidence: {probability[0]*100:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box" style="background: linear-gradient(45deg, #ff416c, #ff4b2b); color: white;">
                                🚨 SPAM DETECTED!<br>
                                <small>Confidence: {probability[1]*100:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show probability breakdown
                        st.markdown("### 📊 Confidence Analysis")
                        col_prob1, col_prob2 = st.columns(2)
                        with col_prob1:
                            st.metric("Safe Probability", f"{probability[0]*100:.1f}%")
                        with col_prob2:
                            st.metric("Spam Probability", f"{probability[1]*100:.1f}%")
            else:
                st.error("❌ Please upload and load model files first!")
    
    with col2:
        st.markdown("### 🎯 Quick Test Samples")
        st.markdown("Click any sample to test the classifier:")
        
        # Create sample buttons
        for title, message in SAMPLE_MESSAGES.items():
            if st.button(f"**{title}**", key=f"sample_{title}", use_container_width=True):
                st.session_state.sample_text = message
                st.rerun()  # Refresh to update the text area
        
        # File upload for custom model
        st.markdown("### 📁 Upload Custom Model")
        
        uploaded_model = st.file_uploader("Upload spam.pkl", type=['pkl'], key="model_upload")
        uploaded_vectorizer = st.file_uploader("Upload vectorizer.pkl", type=['pkl'], key="vectorizer_upload")
        
        if uploaded_model and uploaded_vectorizer:
            if st.button("🔄 Load Uploaded Model", use_container_width=True):
                with st.spinner('🔄 Loading and verifying custom model...'):
                    model, cv, loaded = load_custom_model(uploaded_model, uploaded_vectorizer)
                    if loaded:
                        st.session_state.model = model
                        st.session_state.cv = cv
                        st.session_state.model_loaded = True
                        st.success("✅ Custom model loaded and verified successfully!")
                        
                        # Test the model with a sample
                        test_result, test_prob = classify_text("test email", model, cv)
                        if test_result is not None:
                            st.info(f"✅ Model verification successful! Ready for classification.")
        
        # Model status
        st.markdown("### 🔧 Model Status")
        if st.session_state.model_loaded:
            st.success("✅ Model is loaded and ready!")
            
            # Test the current model with a simple prediction
            if st.button("🧪 Test Model", use_container_width=True):
                test_text = "This is a test email"
                pred, prob = classify_text(test_text, st.session_state.model, st.session_state.cv)
                if pred is not None:
                    status = "HAM" if pred == 0 else "SPAM"
                    st.info(f"Test result: **{status}** (Confidence: {max(prob)*100:.1f}%)")
        else:
            st.error("❌ No model loaded")
    
    # Clean footer without the features section
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Made with ❤️ using Streamlit | SpamGuard AI v2.0"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()