import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_movielens, create_mappings, train_test_split_ratings
from src.matrix_factorization import MatrixFactorization
from src.recommend import get_top_n_recommendations

# Load MovieLens ratings and movies
ratings, movies = load_movielens()
ratings.shape, movies.shape
