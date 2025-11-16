import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_movielens(path_ratings="data/ratings.csv", path_movies="data/movies.csv"):
    ratings = pd.read_csv(path_ratings)
    movies = pd.read_csv(path_movies)
    return ratings, movies

def create_mappings(ratings):
    unique_users = ratings['userId'].unique()
    unique_movies = ratings['movieId'].unique()
    user2idx = {u: i for i, u in enumerate(unique_users)}
    idx2user = {i: u for u, i in user2idx.items()}
    movie2idx = {m: i for i, m in enumerate(unique_movies)}
    idx2movie = {i: m for m, i in movie2idx.items()}
    return user2idx, idx2user, movie2idx, idx2movie

def build_rating_matrix(ratings, user2idx, movie2idx):
    n_users = len(user2idx)
    n_movies = len(movie2idx)
    R = np.zeros((n_users, n_movies), dtype=np.float32)
    for row in ratings.itertuples():
        u = user2idx[row.userId]
        m = movie2idx[row.movieId]
        R[u, m] = row.rating
    return R

def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)
