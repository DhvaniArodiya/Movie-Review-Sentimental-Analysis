import pickle as pk
import streamlit as st
import os

# Get absolute path (safe for Streamlit)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model and TF-IDF vectorizer
model = pk.load(open(os.path.join(BASE_DIR, 'model.pk'), 'rb'))
tfidf = pk.load(open(os.path.join(BASE_DIR, 'tfidf.pk'), 'rb'))

# UI
st.title("Movie Review Sentiment Analysis")
review = st.text_input("Enter Movie Review")

if st.button("Predict"):
    review_vector = tfidf.transform([review])   # ✅ ONLY transform
    result = model.predict(review_vector)

    if result[0] == 0:
        st.success("Negative Review")
    else:
        st.success("Positive Review")
