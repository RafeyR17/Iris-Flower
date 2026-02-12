import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered",
)

# --- Custom Styling ---
st.markdown("""
    <style>
    /* Premium Look and Feel */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stSlider > div > div > div > div {
        color: #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 2rem;
    }
    .species-name {
        font-size: 2.5rem;
        font-weight: 800;
        margin-top: 1rem;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    model = joblib.load('iris_model.joblib')
    scaler = joblib.load('scaler.joblib')
    iris = load_iris()
    return model, scaler, iris.target_names

try:
    model, scaler, target_names = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- App Header ---
st.title("🌸 Iris Flower Classifier")
st.markdown("""
    Predict the species of an Iris flower based on its measurements. 
    This application uses a high-performance **Support Vector Machine (SVM)** model trained on the classic Iris dataset.
""")

# --- Sidebar / Controls ---
st.sidebar.header("Input Features")
st.sidebar.markdown("Adjust the sliders to set the flower's dimensions.")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3, step=0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, step=0.1)

# --- Prediction Logic ---
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
species = target_names[prediction].capitalize()

# --- Main Dashboard ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Selected Dimensions")
    data = {
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width]
    }
    st.table(pd.DataFrame(data))

with col2:
    st.subheader("Prediction Result")
    st.markdown(f"""
        <div class="prediction-card">
            <p style="font-size: 1.1rem; color: #aaa;">Predicted Species</p>
            <div class="species-name">{species}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Visual Context (Optional) ---
st.divider()
st.subheader("Dataset Insights")
st.image('eda_pairplot.png', caption="Feature Relationships (from Training EDA)")

# --- Footer ---
st.markdown("""
    ---
    Built with ❤️ using Streamlit and Scikit-Learn.
""")
