import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open('model/feature-extraction.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load models from HDF5
@st.cache(allow_output_mutation=True)
def load_models():
    model_lstm = load_model('model/model_lstm.h5')
    model_cnn = load_model('model/model_cnn.h5')
    model_bilstm = load_model('model/model_bilstm.h5')
    return model_lstm, model_cnn, model_bilstm

tokenizer = load_tokenizer()
model_lstm, model_cnn, model_bilstm = load_models()

# Preprocessing
def preprocess(text, tokenizer, max_len=100):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    return padded

# UI
st.title("Deteksi Ujaran Kebencian Bahasa Banjar - Multi Model")
text = st.text_area("Masukkan Kalimat:", "Contoh kalimat di sini...")

if st.button("Klasifikasikan"):
    if text.strip():
        x = preprocess(text, tokenizer)

        pred_lstm = model_lstm.predict(x)[0][0]
        pred_cnn = model_cnn.predict(x)[0][0]
        pred_bilstm = model_bilstm.predict(x)[0][0]

        hard_preds = [int(pred >= 0.5) for pred in [pred_lstm, pred_cnn, pred_bilstm]]
        soft_vote = np.mean([pred_lstm, pred_cnn, pred_bilstm])
        hard_vote = round(np.mean(hard_preds))

        labels = {0: "Bukan Ujaran Kebencian", 1: "Ujaran Kebencian"}

        st.subheader("Hasil Prediksi per Model")
        st.write(f"LSTM: {labels[hard_preds[0]]} ({pred_lstm:.2f})")
        st.write(f"CNN: {labels[hard_preds[1]]} ({pred_cnn:.2f})")
        st.write(f"BiLSTM: {labels[hard_preds[2]]} ({pred_bilstm:.2f})")

        st.subheader("Ensemble Voting")
        st.info(f"Soft Voting: {labels[int(soft_vote >= 0.5)]} (avg prob: {soft_vote:.2f})")
        st.warning(f"Hard Voting: {labels[hard_vote]} (mayoritas label)")
    else:
        st.warning("Masukkan teks terlebih dahulu.")
