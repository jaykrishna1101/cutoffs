import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_color = "light"
st.set_page_config(
    page_title="Cutoff Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# Cache model loading for performance
@st.cache_resource
def load_model():
    model_path = 'best_random_forest_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

st.title("🎓 Cutoff Score Prediction App")
st.write("""
This interactive app uses a trained machine learning model to predict university cutoff scores based on your selections.
""")

pipeline = load_model()

if pipeline is None:
    st.warning("⚠️ 'best_random_forest_model.joblib' not found. Please upload the model file below:")
    uploaded_file = st.file_uploader("Upload Model (.joblib)", type=["joblib"])
    if uploaded_file is not None:
        with open("best_random_forest_model.joblib", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Model uploaded and ready!")
        pipeline = joblib.load("best_random_forest_model.joblib")

if pipeline is not None:
    st.subheader("Enter Selection Details")
    
    # Feature input elements
    col1, col2 = st.columns(2)
    
    with col1:
        university_id = st.text_input("University ID", value="banaras-hindu-university")
        university = st.text_input("University Name", value="Banaras Hindu University")
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2024, step=1)
        programme_id = st.text_input("Programme ID", value="banaras-hindu-university-b-a-honours")
        programme = st.text_input("Programme Name", value="B.A. (Honours)")
        college = st.text_input("College", value="Arya Mahila PG College")
        variant = st.text_input("Variant", value="Ancient Indian History Culture & Archaeology")
        
    with col2:
        round_val = st.text_input("Round", value="1")
        category = st.text_input("Category", value="UR")
        seat_type = st.text_input("Seat Type", value="general")
        figure_kind = st.text_input("Figure Kind", value="score")
        basis = st.text_input("Basis", value="official")
        source_id = st.text_input("Source ID", value="banaras-hindu-university-2024-r1")
        source_page = st.number_input("Source Page", min_value=1.0, max_value=1000.0, value=1.0, step=1.0)

    # Compile into a DataFrame matching model expectation
    input_data = pd.DataFrame({
        'university_id': [university_id],
        'university': [university],
        'year': [year],
        'programme_id': [programme_id],
        'programme': [programme],
        'college': [college],
        'variant': [variant],
        'round': [round_val],
        'category': [category],
        'seat_type': [seat_type],
        'figure_kind': [figure_kind],
        'basis': [basis],
        'source_id': [source_id],
        'source_page': [source_page]
    })
    
    if st.button("Predict Cutoff Score", type="primary"):
        try:
            prediction = pipeline.predict(input_data)[0]
            st.balloons()
            st.metric(label="Predicted Cutoff Score", value=f"{prediction:.2f}")
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
