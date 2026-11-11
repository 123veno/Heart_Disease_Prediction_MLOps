import requests
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.caption("Logistic Regression | Educational machine-learning prototype")
st.warning("This application is for educational purposes only. It cannot diagnose heart disease or replace professional medical advice.")

API_URL = st.sidebar.text_input("Prediction API URL", "http://api:8000")
# API_URL = st.sidebar.text_input("Prediction API URL", "http://127.0.0.1:8000")
with st.form("patient_form"):
    st.subheader("Patient attributes")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 55)
        sex = st.selectbox("Sex (0=female, 1=male)", [0, 1])
        cp = st.selectbox("Chest pain type (1–4)", [1, 2, 3, 4])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 40, 300, 140)
        chol = st.number_input("Cholesterol (mg/dl)", 50, 700, 220)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    with c2:
        thalach = st.number_input("Maximum heart rate achieved", 40, 250, 150)
        exang = st.selectbox("Exercise-induced angina", [0, 1])
        oldpeak = st.number_input("ST depression (oldpeak)", -5.0, 15.0, 1.5, step=0.1)
        slope = st.selectbox("Slope of peak exercise ST segment", [1, 2, 3])
        ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal value", [3, 6, 7])
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }
    try:
        response = requests.post(f"{API_URL.rstrip('/')}/predict", json=payload, timeout=15)
        if response.ok:
            result = response.json()
            if result["prediction"] == 1:
                st.error(result["label"])
            else:
                st.success(result["label"])
            st.metric("Model-estimated probability for class 1", f"{result['probability']*100:.1f}%")
            st.caption(result["disclaimer"])
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
    except requests.RequestException as exc:
        st.error(f"Cannot connect to prediction API: {exc}. Check that the API service is running.")
