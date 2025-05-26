import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    model_lstm = load_model('model/model_lstm.h5')
    model_cnn = load_model('model/model_cnn.h5')
    model_bilstm = load_model('model/model_bilstm.h5')
    return model_lstm, model_cnn, model_bilstm

# Inisialisasi
tokenizer = load_tokenizer()
model_lstm, model_cnn, model_bilstm = load_models()

# Preprocessing
def preprocess(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded

# Fungsi tampil hasil dengan warna
def tampilkan_hasil(label, sumber=""):
    warna = "green" if label == 0 else "red"
    teks = "Bukan Ujaran Kebencian" if label == 0 else "Ujaran Kebencian"
    st.markdown(f"<h4 style='color:{warna}'>{sumber}: {teks}</h4>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Pilih Model Klasifikasi")
model_option = st.sidebar.radio(
    "Model yang digunakan:",
    ("LSTM", "CNN", "BiLSTM", "Ensemble Voting")
)

# UI utama\\
st.title("B-HSD")
st.title("Banjar Hate Speech Detection")
text = st.text_area("Masukkan Kalimat:")

if st.button("Klasifikasikan"):
    if text.strip():
        x = preprocess(text, tokenizer)

        # Prediksi semua model
        pred_lstm = model_lstm.predict(x)[0][0]
        pred_cnn = model_cnn.predict(x)[0][0]
        pred_bilstm = model_bilstm.predict(x)[0][0]

        if model_option == "LSTM":
            hasil = int(pred_lstm >= 0.5)
            tampilkan_hasil(hasil, "Hasil Deteksi (LSTM)")

        elif model_option == "CNN":
            hasil = int(pred_cnn >= 0.5)
            tampilkan_hasil(hasil, "Hasil Deteksi (CNN)")

        elif model_option == "BiLSTM":
            hasil = int(pred_bilstm >= 0.5)
            tampilkan_hasil(hasil, "Hasil Deteksi (BiLSTM)")

        elif model_option == "Ensemble Voting":
            # Soft voting
            soft_vote = np.mean([pred_lstm, pred_cnn, pred_bilstm])
            soft_label = int(soft_vote >= 0.5)

            # Hard voting
            hard_preds = [int(pred >= 0.5) for pred in [pred_lstm, pred_cnn, pred_bilstm]]
            hard_vote = round(np.mean(hard_preds))

            tampilkan_hasil(soft_label, "Soft Voting")
            tampilkan_hasil(hard_vote, "Hard Voting (Mayoritas Model)")

    else:
        st.warning("Masukkan teks terlebih dahulu.")
