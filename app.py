import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- Custom Focal Loss ----
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
        return alpha * (1-p_t)**gamma * bce
    return loss

# ---- Load Model ----
model = load_model(
    "hybrid_model.h5",
    custom_objects={"loss": focal_loss()}
)

# ---- Auto sequence length ----
SEQ_LEN = model.input_shape[0][1]

# ---- Best threshold ----
BEST_THRESHOLD = 0.7039

# ---- UI ----
st.title("🧬 CRISPR Off-Target Risk Predictor")

# ---- Threshold Selection ----
st.subheader("⚙ Choose Risk Sensitivity")

threshold_option = st.selectbox(
    "Select Threshold Mode",
    ["Best Threshold (Recommended)", "Safe Mode (0.3 - High Recall)", "Custom"]
)

if threshold_option == "Best Threshold (Recommended)":
    selected_threshold = BEST_THRESHOLD
elif threshold_option == "Safe Mode (0.3 - High Recall)":
    selected_threshold = 0.3
else:
    selected_threshold = st.slider("Custom Threshold", 0.0, 1.0, 0.5)

# ---- Encoding ----
def one_hot_2d(seq):
    mapping = {
        'A':[1,0,0,0],
        'T':[0,1,0,0],
        'G':[0,0,1,0],
        'C':[0,0,0,1]
    }
    return np.array([mapping.get(i,[0,0,0,0]) for i in seq])

# ---- Feature Functions ----
def mismatch_count(sg, off):
    return sum(1 for a,b in zip(sg,off) if a!=b)

def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

# ---- Highlight Mismatch ----
def highlight_mismatch(sg, off):
    result = ""
    for a, b in zip(sg, off):
        if a == b:
            result += a
        else:
            result += f"[{b}]"
    return result

# ---- PAM Check ----
def check_pam(seq):
    return "✅ Valid PAM (NGG detected)" if seq.endswith("GG") else "❌ No PAM (NGG missing)"

# =========================
# 🔍 SINGLE PREDICTION
# =========================
st.subheader("🔍 Single Prediction")

sgRNA = st.text_input("Enter sgRNA sequence")
off_seq = st.text_input("Enter Off-target sequence")

if st.button("Predict"):

    if len(sgRNA) == 0 or len(off_seq) == 0:
        st.warning("Please enter both sequences")
        st.stop()

    if sgRNA.upper() == off_seq.upper():
        st.error("🚨 Perfect match → HIGH RISK (On-target)")
        st.stop()

    # ---- Combine ----
    combined = sgRNA.upper() + off_seq.upper()
    seq_encoded = one_hot_2d(combined)

    if seq_encoded.shape[0] != SEQ_LEN:
        st.error(f"Combined sequence length must be {SEQ_LEN}")
        st.stop()

    # ✅ FIX: dtype + reshape
    seq_encoded = seq_encoded.reshape(1, SEQ_LEN, 4).astype(np.float32)

    # ✅ FIX: normalized + float32
    manual_features = np.array([[
        mismatch_count(sgRNA.upper(), off_seq.upper()) / len(sgRNA),
        gc_content(sgRNA.upper())
    ]], dtype=np.float32)

    # ---- Prediction ----
    prob = model.predict([seq_encoded, manual_features], verbose=0)[0][0]

    # ---- Output ----
    st.write(f"### 🔎 Prediction Probability: {prob:.6f}")
    st.write("Mismatch:", mismatch_count(sgRNA, off_seq))
    st.write("GC Content:", gc_content(sgRNA))

    # ---- Confidence ----
    st.subheader("📊 Confidence")
    st.progress(float(prob))
    st.metric("Risk Score", f"{prob:.2f}")

    # ---- Sequence Highlight ----
    st.subheader("🧬 Sequence Comparison")
    st.write("sgRNA:      ", sgRNA.upper())
    st.write("off-target: ", highlight_mismatch(sgRNA.upper(), off_seq.upper()))

    # ---- PAM Check ----
    st.subheader("🧪 PAM Check")
    st.write(check_pam(off_seq.upper()))

    # ---- Decision ----
    if prob > selected_threshold:
        st.error("⚠ High Off-Target Risk")
    else:
        st.success("✅ Low Off-Target Risk")

    st.caption(f"Threshold used: {selected_threshold:.4f}")

# =========================
# 📁 CSV BATCH PREDICTION
# =========================
st.subheader("📁 Batch Prediction (CSV Upload)")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    results = []

    for _, row in df.iterrows():
        sg = row['sgRNA']
        off = row['off_target']

        combined = sg.upper() + off.upper()
        seq_encoded = one_hot_2d(combined)

        if seq_encoded.shape[0] != SEQ_LEN:
            continue

        seq_encoded = seq_encoded.reshape(1, SEQ_LEN, 4).astype(np.float32)

        manual_features = np.array([[
            mismatch_count(sg.upper(), off.upper()) / len(sg),
            gc_content(sg.upper())
        ]], dtype=np.float32)

        prob = model.predict([seq_encoded, manual_features], verbose=0)[0][0]

        risk = "High" if prob > selected_threshold else "Low"

        results.append([sg, off, prob, risk])

    result_df = pd.DataFrame(results, columns=["sgRNA", "Off-target", "Probability", "Risk"])

    st.write(result_df)