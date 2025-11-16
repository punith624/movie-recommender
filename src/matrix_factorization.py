import numpy as np
from tqdm import trange

class MatrixFactorization:
    def __init__(self, n_users, n_items, k=20, lr=0.01, reg=0.02, n_epochs=50, verbose=True):
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = None

    def _init_matrices(self):
        self.P = np.random.normal(scale=1.0/self.k, size=(self.n_users, self.k))
        self.Q = np.random.normal(scale=1.0/self.k, size=(self.n_items, self.k))
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)

    def fit(self, train_data, val_data=None):
        self._init_matrices()
        if len(train_data) == 0:
            raise ValueError("Train data is empty")
        self.mu = np.mean([r for (_, _, r) in train_data])
        history = {'train_rmse': [], 'val_rmse': []}
        for epoch in trange(self.n_epochs, desc='Training'):
            np.random.shuffle(train_data)
            for (u, i, r) in train_data:
                pred = self.predict_single(u, i)
                e_ui = r - pred
                self.bu[u] += self.lr * (e_ui - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e_ui - self.reg * self.bi[i])
                P_u = self.P[u, :].copy()
                Q_i = self.Q[i, :].copy()
                self.P[u, :] += self.lr * (e_ui * Q_i - self.reg * P_u)
                self.Q[i, :] += self.lr * (e_ui * P_u - self.reg * Q_i)
            train_rmse = self.rmse(train_data)
            history['train_rmse'].append(train_rmse)
            if val_data is not None:
                val_rmse = self.rmse(val_data)
                history['val_rmse'].append(val_rmse)
                if self.verbose:
                    print(f"Epoch {len(history['train_rmse'])}: train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")
            else:
                if self.verbose:
                    print(f"Epoch {len(history['train_rmse'])}: train_rmse={train_rmse:.4f}")
        return history

    def predict_single(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u, :].dot(self.Q[i, :].T)

    def full_prediction_matrix(self):
        return self.mu + self.bu[:, np.newaxis] + self.bi[np.newaxis, :] + self.P.dot(self.Q.T)

    def rmse(self, data):
        se = 0.0
        for (u, i, r) in data:
            pred = self.predict_single(u, i)
            se += (r - pred) ** 2
        return (se / len(data)) ** 0.5

    def save(self, path_prefix):
        np.save(path_prefix + '_P.npy', self.P)
        np.save(path_prefix + '_Q.npy', self.Q)
        np.save(path_prefix + '_bu.npy', self.bu)
        np.save(path_prefix + '_bi.npy', self.bi)
        np.save(path_prefix + '_mu.npy', np.array([self.mu]))

    def load(self, path_prefix):
        self.P = np.load(path_prefix + '_P.npy')
        self.Q = np.load(path_prefix + '_Q.npy')
        self.bu = np.load(path_prefix + '_bu.npy')
        self.bi = np.load(path_prefix + '_bi.npy')
        self.mu = float(np.load(path_prefix + '_mu.npy')[0])
