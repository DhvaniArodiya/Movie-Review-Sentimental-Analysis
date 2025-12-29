import streamlit as st
import pickle as pk
import os

st.set_page_config(page_title="Movie Review Sentiment Analysis")

BASE_DIR = os.path.dirname(__file__)

# Debug: show files (remove later)
st.write("Files in app directory:", os.listdir(BASE_DIR))

# Load model and tfidf
model_path = os.path.join(BASE_DIR, "model.pk")
tfidf_path = os.path.join(BASE_DIR, "tfidf.pk")

model = pk.load(open(model_path, "rb"))
tfidf = pk.load(open(tfidf_path, "rb"))

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        review_vec = tfidf.transform([review])
        result = model.predict(review_vec)

        if result[0] == 0:
            st.error("Negative Review 😞")
        else:
            st.success("Positive Review 😊")

