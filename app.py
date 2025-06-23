import streamlit as st
import joblib
import base64

# ---------------------- Page Config ----------------------
st.set_page_config(page_title="🧠 Text Intent Classifier", layout="centered")

# ---------------------- Load Model & Vectorizer ----------------------
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------------- Custom CSS Styling ----------------------
st.markdown("""
    <style>
        .main {
            background-color: #F7F8FC;
        }
        .stButton>button {
            color: white;
            background-color: #6C63FF;
            border-radius: 10px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stTextArea>div>textarea {
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 10px;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- App Title ----------------------
st.markdown("<h1 style='text-align:center; color:#6C63FF;'>🧠 Text Intent Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Classify chat messages or sentences into appropriate categories using a trained Random Forest model</p>", unsafe_allow_html=True)

# ---------------------- Example Prompts ----------------------
st.markdown("#### ✨ Try an Example")
example_texts = [
    "I'm so excited to play Borderlands tonight!",
    "I'm getting on Borderlands and I will murder you all.",
    "Thank you for the help, really appreciate it.",
    "Just quit already, nobody wants you here.",
    "I finished the level. What’s next?"
]

selected_example = st.selectbox("Choose an example or enter your own text below:", [""] + example_texts)

# ---------------------- Text Input ----------------------
user_input = st.text_area("🔍 Enter text for prediction:", selected_example)

# ---------------------- Predict Button ----------------------
if st.button("🚀 Predict"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        st.success(f"🎯 **Predicted Class:** `{prediction}`")
    else:
        st.warning("⚠️ Please enter or select some text above.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px; color: #888;'>Created with ❤️ by Ume Habiba | Powered by Streamlit</div>",
    unsafe_allow_html=True
)
