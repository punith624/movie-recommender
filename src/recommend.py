import numpy as np

def get_top_n_recommendations(model, user_idx, idx2movie, movies_df, train_mask=None, n=10):
    pred = model.full_prediction_matrix()[user_idx]
    if train_mask is not None:
        pred = pred.copy()
        pred[train_mask[user_idx]] = -np.inf
    top_idx = np.argsort(pred)[::-1][:n]
    movie_ids = [idx2movie[i] for i in top_idx]
    df = movies_df[movies_df['movieId'].isin(movie_ids)].copy()
    df["pred_score"] = pred[top_idx]
    return df
