import streamlit as st
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Fungsi untuk menghapus emoji
def remove_emoji(text):
    emoji_pattern = re.compile("["                     
        u"\U0001F600-\U0001F64F"                     
        u"\U0001F300-\U0001F5FF"                     
        u"\U0001F680-\U0001F6FF"                     
        u"\U0001F1E0-\U0001F1FF"                     
        u"\U00002702-\U000027B0"                     
        u"\U000024C2-\U0001F251"                     
        u"\U0001F900-\U0001F9FF"                     
        u"\U0001F1F2-\U0001F1F4"                     
        u"\U0001F1E6-\U0001F1FF"                     
        u"\U0001F681-\U0001F6C5"                     
        u"\U0001F30D-\U0001F567"                     
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Fungsi preprocessing lengkap
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # hapus tanda baca
    text = re.sub(r'\d+', '', text)      # hapus angka
    text = remove_emoji(text)            # hapus emoji
    text = re.sub(r'\s+', ' ', text)     # hapus spasi ganda
    return text.strip()

# Load tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open('model/feature-extraction.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    model_lstm = load_model('model/model_lstm.h5')
    model_cnn = load_model('model/model_cnn.h5')
    model_bilstm = load_model('model/model_bilstm.h5')
    return model_lstm, model_cnn, model_bilstm

# Fungsi preprocessing dan tokenisasi untuk input user
def preprocess_user_input(text, tokenizer, max_len=100):
    clean_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded

# Streamlit UI
st.title("Deteksi Ujaran Kebencian Bahasa Banjar - Multi Model")

text_input = st.text_area("Masukkan Kalimat:", "Contoh: Ulun kada suka ikam")

tokenizer = load_tokenizer()
model_lstm, model_cnn, model_bilstm = load_models()

if st.button("Klasifikasikan"):
    if text_input.strip():
        x = preprocess_user_input(text_input, tokenizer)

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
