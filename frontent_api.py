import streamlit as st
import pandas as pd
import requests

st.set_page_config("CRISPR Off-Target Tool", layout="wide")
st.title("🧬 CRISPR Off-Target Prediction – Advanced Tool")

mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch CSV Upload"])

API_URL = "http://localhost:8000/predict"

# ---------------- SINGLE PREDICTION ----------------
if mode == "Single Prediction":
    sg = st.text_input("sgRNA (20 bp)")
    off = st.text_input("Off-target DNA (20 bp)")

    if st.button("Predict"):
        res = requests.post(
            API_URL,
            json={"sgRNA": sg, "off_seq": off}
        ).json()

        st.metric("Off-Target Probability", f"{res['probability']:.3f}")
        st.write("Risk Level:", res["risk"])

# ---------------- BATCH CSV UPLOAD ----------------
else:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Uploaded Data Preview")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            res = requests.post(
                API_URL + "/batch",
                files={"file": file}
            ).json()

            out = pd.DataFrame(res)
            st.success("Prediction Completed")
            st.dataframe(out)

            st.download_button(
                "Download Results",
                out.to_csv(index=False),
                file_name="crispr_predictions.csv"
            )

