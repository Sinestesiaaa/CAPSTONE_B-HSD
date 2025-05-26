import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer dari file tokenizer.pkl
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load model-model dari file HDF5
@st.cache(allow_output_mutation=True)
def load_models():
    model_lstm = load_model('model/model_lstm.h5')
    model_cnn = load_model('model/model_cnn.h5')
    model_bilstm = load_model('model/model_bilstm.h5')
    return model_lstm, model_cnn, model_bilstm

# Inisialisasi
tokenizer = load_tokenizer()
model_lstm, model_cnn, model_bilstm = load_models()

# Fungsi preprocessing
def preprocess(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded

# Label hasil klasifikasi
labels = {0: "Bukan Ujaran Kebencian", 1: "Ujaran Kebencian"}

# Sidebar untuk memilih model
st.sidebar.title("Pilih Model Klasifikasi")
model_option = st.sidebar.radio(
    "Model yang digunakan:",
    ("LSTM", "CNN", "BiLSTM", "Ensemble Voting")
)

# UI utama
st.title("Deteksi Ujaran Kebencian Bahasa Banjar")
text = st.text_area("Masukkan Kalimat:")

if st.button("Klasifikasikan"):
    if text.strip():
        x = preprocess(text, tokenizer)

        pred_lstm = model_lstm.predict(x)[0][0]
        pred_cnn = model_cnn.predict(x)[0][0]
        pred_bilstm = model_bilstm.predict(x)[0][0]

        if model_option == "LSTM":
            hasil = int(pred_lstm >= 0.5)
            st.success(f"Hasil Deteksi (LSTM): {labels[hasil]}")

        elif model_option == "CNN":
            hasil = int(pred_cnn >= 0.5)
            st.success(f"Hasil Deteksi (CNN): {labels[hasil]}")

        elif model_option == "BiLSTM":
            hasil = int(pred_bilstm >= 0.5)
            st.success(f"Hasil Deteksi (BiLSTM): {labels[hasil]}")

        elif model_option == "Ensemble Voting":
            hard_preds = [int(pred >= 0.5) for pred in [pred_lstm, pred_cnn, pred_bilstm]]
            hard_vote = round(np.mean(hard_preds))
            st.success(f"Hasil Deteksi (Ensemble Hard Voting): {labels[hard_vote]}")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
