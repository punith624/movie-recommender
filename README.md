# 🎬 Movie Recommender — Matrix Factorization (From Scratch)

A clean, portfolio-ready implementation of a Matrix Factorization-based movie recommender trained on MovieLens-100K. Includes from-scratch SGD optimization, user/item biases, evaluation, hyperparameter tuning, and latent-factor visualization (PCA & t-SNE). Comes with polished notebooks and a Streamlit demo for interactive recommendations.

---

## 📸 Quick Demo / Screenshot

![Streamlit Demo](assets/demo.png)

---

## 🚀 Key Features

- Matrix Factorization built from scratch using NumPy  
- User & item bias terms included  
- Stochastic Gradient Descent (SGD) optimizer  
- Train/Validation RMSE tracking  
- Hyperparameter tuning (grid search)  
- Latent factor visualization (PCA & t-SNE)  
- Interactive Streamlit demo app  
- Modular and clean code inside `src/`  

---

## 📝 Resume-Ready Summary

**One-liner:**  
Built a Matrix Factorization movie recommender from scratch (NumPy) on MovieLens-100K with SGD, hyperparameter tuning, PCA/t-SNE visualization, and a Streamlit demo.

**Two-liner:**  
Implemented a complete Matrix Factorization recommender using NumPy with SGD optimization, user/item biases, and RMSE-based validation. Developed hyperparameter tuning pipelines, latent-factor visualizations, and an interactive Streamlit interface with modular code and well-documented notebooks.

---

## 📚 Table of Contents

- [Quick Demo / Screenshot](#-quick-demo--screenshot)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Notebooks](#-notebooks)
- [Project Structure](#-project-structure)
- [Implementation Notes](#-implementation-notes)
- [Presentation Tips](#-presentation-tips)
- [Contributing](#-contributing)
- [Contact](#-contact)

---

## 🔧 Installation

### Requirements
- Python 3.8+
- Recommended: virtual environment

### Setup
```bash
git clone https://github.com/punith624/movie-recommender.git
cd movie-recommender

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt


## Usage 

**Train the model**

```bash
python main.py --train --k 30 --lr 0.007 --reg 0.02 --epochs 30
```


# Launch the Streamlit App
streamlit run app.py


# Open in your browser:
👉 http://localhost:8501

# Run the Jupyter Notebooks

Open the .ipynb files in the notebooks/ directory.

## Notebooks
Notebook	Description
01_professional_notebook.ipynb	Full walkthrough: math, EDA, MF training, evaluation
02_hyperparam_tuning.ipynb	Grid search for k, learning rate, regularization
03_latent_viz.ipynb	PCA & t-SNE visualization of latent factors
# Project Structure
movie-recommender/
├── assets/
│   └── demo.png
├── data/
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── matrix_factorization.py
│   ├── recommend.py
│   └── evaluation.py
├── main.py
├── app.py
├── requirements.txt
└── README.md

# Implementation Notes
# Matrix Factorization Model

The predicted rating is:

𝑟
^
𝑢
𝑖
=
𝜇
+
𝑏
𝑢
+
𝑏
𝑖
+
𝑝
𝑢
⊤
𝑞
𝑖
r
^
ui
	​

=μ+b
u
	​

+b
i
	​

+p
u
⊤
	​

q
i
	​


Where:

𝜇
μ: global mean

𝑏
𝑢
b
u
	​

: user bias

𝑏
𝑖
b
i
	​

: item bias

𝑝
𝑢
p
u
	​

, 
𝑞
𝑖
q
i
	​

: latent factor vectors

 # Optimization

SGD with L2 regularization

Optionally shuffle per epoch

Supports mini-batch extension

# Evaluation Metrics

RMSE for both train and validation

Logged and plotted per epoch

# Visualization

PCA → global structure of the latent space

t-SNE → clustering of similar movies

Helps interpret genre patterns & similarities

## Presentation Tips

Show RMSE vs Epochs to demonstrate training stability

Briefly explain the MF formula (bias terms + dot product)

Present Top-N recommendations for a user

Include PCA / t-SNE scatterplots

Mention training time (~3 min on ML-100K)

Highlight modular Python architecture

## Recommended .gitignore
venv/
__pycache__/
*.pyc
data/
models/
.env

 ## Contributing

Pull requests and suggestions are welcome!

### Contact

Punith Kumar
GitHub: https://github.com/punith624

Email: kumarpunith6864@gmail.com



