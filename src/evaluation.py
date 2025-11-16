import numpy as np

def rmse_from_matrix(R_true_triplets, prediction_matrix):
    se = 0.0
    for (u, i, r) in R_true_triplets:
        se += (r - prediction_matrix[u, i]) ** 2
    return (se / len(R_true_triplets)) ** 0.5
