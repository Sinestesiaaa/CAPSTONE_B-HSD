import joblib
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Membaca model dengan joblib
hate_speech_detection = joblib.load('ensemble-method.pkl')
cv = joblib.load('feature-extraction.pkl')

# Judul web
st.title("Deteksi Ujaran Kebencian Bahasa Banjar")

# Input teks dari pengguna
text = st.text_area("Masukkan Kalimat dalam Bahasa Banjar", "Masukkan Kalimat di sini")

# Label klasifikasi
prediction_labels = {'Ujaran Kebencian': 1, 'Bukan Ujaran Kebencian': 0}

if st.button("Klasifikasikan"):
    if text.strip():  # Memeriksa apakah teks tidak kosong setelah tombol diklik
        vect_text = cv.transform([text])

        # Melakukan prediksi
        prediction = hate_speech_detection.predict(vect_text)[0]  # Ambil nilai prediksi pertama

        final_result = get_key(prediction, prediction_labels)

        if final_result == 'Bukan Ujaran Kebencian':
            st.success(f"Kalimat Termasuk: {final_result}")
        elif final_result == 'Ujaran Kebencian':
            st.error(f"Kalimat Termasuk: {final_result}")
    else:
        st.warning("Masukkan teks sebelum melakukan klasifikasi.")