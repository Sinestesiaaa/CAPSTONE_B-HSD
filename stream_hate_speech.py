import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load tokenizer ---
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open('model/feature-extraction.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# --- Load models from .h5 files ---
@st.cache(allow_output_mutation=True)
def load_models():
    model_lstm = load_model('model/model_lstm.h5')
    model_cnn = load_model('model/model_cnn.h5')
    model_bilstm = load_model('model/model_bilstm.h5')
    return model_lstm, model_cnn, model_bilstm

# --- Load all assets ---
tokenizer = load_tokenizer()
model_lstm, model_cnn, model_bilstm = load_models()
max_length = 100  # Sesuaikan dengan saat training

# --- Preprocessing function ---
def preprocess(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded

# --- UI ---
st.title("🗣️ Deteksi Ujaran Kebencian Bahasa Banjar - Multi Model")

text = st.text_area("Masukkan Kalimat dalam Bahasa Banjar:", "Tambahkan teks di sini")

if st.button("Klasifikasikan"):
    if text.strip():
        x = preprocess(text, tokenizer, max_length)

        # --- Individual Predictions ---
        pred_lstm = model_lstm.predict(x)[0][0]
        pred_cnn = model_cnn.predict(x)[0][0]
        pred_bilstm = model_bilstm.predict(x)[0][0]

        # --- Label Mapping ---
        labels = {0: "Bukan Ujaran Kebencian", 1: "Ujaran Kebencian"}

        # --- Hard predictions ---
        hard_preds = [int(p >= 0.5) for p in [pred_lstm, pred_cnn, pred_bilstm]]

        # --- Ensemble Voting ---
        soft_vote = np.mean([pred_lstm, pred_cnn, pred_bilstm])
        hard_vote = round(np.mean(hard_preds))

        # --- Output per model ---
        st.subheader("🔍 Hasil Prediksi per Model")
        st.write(f"**LSTM**: {labels[hard_preds[0]]} (Prob: `{pred_lstm:.2f}`)")
        st.write(f"**CNN**: {labels[hard_preds[1]]} (Prob: `{pred_cnn:.2f}`)")
        st.write(f"**BiLSTM**: {labels[hard_preds[2]]} (Prob: `{pred_bilstm:.2f}`)")

        # --- Output ensemble ---
        st.subheader("🧠 Ensemble Voting")
        st.info(f"**Soft Voting**: {labels[int(soft_vote >= 0.5)]} (Rata-rata Probabilitas: `{soft_vote:.2f}`)")
        st.warning(f"**Hard Voting**: {labels[hard_vote]} (Mayoritas Label)")
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu.")
