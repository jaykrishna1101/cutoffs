# University Cutoff Score Predictor Web App

A lightweight Streamlit web application deployed with a trained Random Forest regressor to predict university admission cutoffs.

## Local Installation

1. Clone this repository or download the package files.
2. Create a virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application locally:
   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud Deployment

1. Push all files including `best_random_forest_model.joblib` to a public repository on GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in.
3. Click **New app**, select your repository, branch, and set the Main file path to `app.py`.
4. Click **Deploy**!
