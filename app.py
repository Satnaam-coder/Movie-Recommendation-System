import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load files
df = pickle.load(open('df.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
indices = pickle.load(open('indices.pkl', 'rb'))

# Recommendation function
def recommend(title):
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie:",
    df['title'].values
)

if st.button("Recommend"):
    results = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in results:
        st.write(movie)