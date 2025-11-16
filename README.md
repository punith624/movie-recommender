# Movie Recommender — Matrix Factorization (From Scratch)

[![Repo Size](https://img.shields.io/github/repo-size/punith624/movie-recommender)](https://github.com/punith624/movie-recommender)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/punith624/movie-recommender)](LICENSE)
[![Issues](https://img.shields.io/github/issues-raw/punith624/movie-recommender)](https://github.com/punith624/movie-recommender/issues)

> A clean, portfolio-ready implementation of a Matrix Factorization movie recommender trained on MovieLens-100K.  
> Includes from-scratch SGD implementation with user/item biases, evaluation, hyperparameter tuning, and latent-factor visualizations (PCA & t-SNE). Comes with notebooks and a Streamlit demo.

---

# Quick demo / Screenshot
*(Replace the image with your own screenshot saved in `/assets/` or `/docs/`)*

![Project Screenshot](assets/screenshot-placeholder.png)

---

# Key features
- Matrix Factorization implemented from scratch (NumPy) with user & item biases
- SGD training with regularization and train/validation RMSE tracking
- Hyperparameter tuning notebook (grid search)
- Latent-factor visualization (PCA & t-SNE) for model interpretability
- Streamlit app for interactive exploration and recommendations
- Clean, modular `src/` structure ready for extension

---

# Resume-ready summary (copy/paste)
**One-line:** Built a Matrix Factorization movie recommender from scratch (NumPy) on MovieLens-100K; implemented SGD with user/item biases, performed hyperparameter tuning, and produced PCA/t-SNE visualizations — reproducible code and a Streamlit demo included.

**Two-line:** Developed a from-scratch Matrix Factorization recommender (NumPy) trained on MovieLens-100K, implementing SGD with user & item biases and RMSE-based validation. Ran hyperparameter grid search, saved model artifacts, and visualized latent factors (PCA & t-SNE); delivered an interactive Streamlit demo and well-documented notebooks.

---

# Table of contents
- [Quick demo / Screenshot](#quick-demo--screenshot)  
- [Key features](#key-features)  
- [Getting started](#getting-started)  
- [Usage](#usage)  
- [Notebooks](#notebooks)  
- [Project structure](#project-structure)  
- [Implementation notes](#implementation-notes)  
- [Tips for presenting this project](#tips-for-presenting-this-project)  
- [Contributing & License](#contributing--license)  
- [Contact](#contact)

---

# Getting started

## Requirements
- Python 3.8+  
- Recommended: create a virtual environment

## Install
```bash
# Clone (if not already)
git clone https://github.com/punith624/movie-recommender.git
cd movie-recommender

# Create venv (Windows example)
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
