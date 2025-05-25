import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load tokenizer ---
with open('model/feature-extraction.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 100  # Sesuaikan dengan saat training

# --- Load individual models (HDF5) ---
@st.cache_resource
def load_models():
    model_lstm = tf.keras.models.load_model('model/model_lstm.h5')
    model_cnn = tf.keras.models.load_model('model/model_cnn.h5')
    model_bilstm = tf.keras.models.load_model('model/model_bilstm.h5')
    return model_lstm, model_cnn, model_bilstm

model_lstm, model_cnn, model_bilstm = load_models()

# --- UI ---
st.title("Deteksi Ujaran Kebencian Bahasa Banjar - Multi Model")

text = st.text_area("Masukkan Kalimat dalam Bahasa Banjar", "Masukkan Kalimat di sini")

labels = {0: 'Bukan Ujaran Kebencian', 1: 'Ujaran Kebencian'}

def preprocess(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

if st.button("Klasifikasikan"):
    if text.strip():
        processed = preprocess(text)

        # Individual predictions
        pred_lstm = model_lstm.predict(processed)[0][0]
        pred_cnn = model_cnn.predict(processed)[0][0]
        pred_bilstm = model_bilstm.predict(processed)[0][0]

        # Konversi ke label (0 atau 1)
        hard_lstm = int(pred_lstm >= 0.5)
        hard_cnn = int(pred_cnn >= 0.5)
        hard_bilstm = int(pred_bilstm >= 0.5)

        # Soft Voting (rata-rata probabilitas)
        soft_vote = np.mean([pred_lstm, pred_cnn, pred_bilstm])
        soft_label = int(soft_vote >= 0.5)

        # Hard Voting (mayoritas label)
        hard_vote = int(round(np.mean([hard_lstm, hard_cnn, hard_bilstm])))

        # --- Tampilkan hasil ---
        st.subheader("Hasil Prediksi Per Model")
        st.write(f"LSTM: {labels[hard_lstm]} ({pred_lstm:.2f})")
        st.write(f"CNN: {labels[hard_cnn]} ({pred_cnn:.2f})")
        st.write(f"BiLSTM: {labels[hard_bilstm]} ({pred_bilstm:.2f})")

        st.subheader("Ensemble Voting")
        st.info(f"Soft Voting: {labels[soft_label]} (avg prob: {soft_vote:.2f})")
        st.warning(f"Hard Voting: {labels[hard_vote]} (mayoritas label)")

    else:
        st.warning("Masukkan teks sebelum melakukan klasifikasi.")
