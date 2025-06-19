import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Charger modèle et tokenizer
model = tf.keras.models.load_model("modele_lstm_sms.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Paramètres
max_len = 50
nlp = spacy.load("fr_core_news_sm")

# Fonction de nettoyage
def nettoyer_spacy(texte):
    doc = nlp(texte.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token) > 1
    ]
    return " ".join(tokens)

# Interface Streamlit
st.title("📱 Détecteur d'agressivité dans un SMS")
st.write("Analyse instantanée du ton d’un message texte : agressif 😠 ou non agressif 😊")

# Entrée utilisateur
sms_input = st.text_area("✉️ Entre ton message ici :", height=100)

if st.button("🔍 Analyser"):
    if sms_input.strip():
        clean = nettoyer_spacy(sms_input)
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        proba = model.predict(padded)[0][0]
        st.markdown("### Résultat")

        if proba >= 0.5:
            st.error(f"😠 Message détecté comme **agressif** (score : {proba:.2f})")
        else:
            st.success(f"😊 Message détecté comme **non agressif** (score : {proba:.2f})")
    else:
        st.warning("Merci de saisir un message avant d’analyser.")
