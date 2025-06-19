import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger le tokenizer et le modèle
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model("modele_lstm_sms.h5")

# Paramètres
max_len = 50

# Titre de l’app
st.title("🔍 Détecteur d’agressivité dans un SMS")

# Input utilisateur
sms = st.text_area("Écris ton message ici :")

# Prédiction
if st.button("Analyser"):
    seq = tokenizer.texts_to_sequences([sms])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    proba = model.predict(pad)[0][0]
    
    if proba >= 0.5:
        st.error(f"Agressif 😠 (score : {proba:.2f})")
    else:
        st.success(f"Non agressif 😊 (score : {proba:.2f})")
