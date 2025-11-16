import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_movielens, create_mappings
from src.matrix_factorization import MatrixFactorization
from src.recommend import get_top_n_recommendations
import os

st.title("Movie Recommender (Matrix Factorization)")

ratings, movies = load_movielens()
user2idx, idx2user, movie2idx, idx2movie = create_mappings(ratings)

model_path_prefix = "models/mf_model"
if not os.path.exists(model_path_prefix + "_P.npy"):
    st.warning("Model not found. Run 'python main.py' to train and create models/ files.")
else:
    mf = MatrixFactorization(1,1)
    mf.load(model_path_prefix)

uid = st.number_input("Enter original userId", min_value=int(ratings.userId.min()), max_value=int(ratings.userId.max()), value=int(ratings.userId.min()))
if st.button("Recommend"):
    if uid not in user2idx:
        st.error("User not in dataset")
    else:
        uidx = user2idx[uid]
        # build train mask
        train_mask = np.zeros((len(user2idx), len(movie2idx)), dtype=bool)
        for r in ratings.itertuples():
            train_mask[user2idx[r.userId], movie2idx[r.movieId]] = True
        recs = get_top_n_recommendations(mf, uidx, idx2movie, movies, train_mask=train_mask, n=10)
        st.table(recs[['title','pred_score']].head(10))
