🎬 Movie Recommender — Matrix Factorization (From Scratch)








A clean, portfolio-ready implementation of a Matrix Factorization-based movie recommender trained on MovieLens-100K.
Includes from-scratch SGD optimization, user/item biases, evaluation, hyperparameter tuning, and latent-factor visualization (PCA & t-SNE).
Comes with polished notebooks and a Streamlit demo for interactive recommendations.

📸 Quick Demo / Screenshot

🚀 Key Features

Matrix Factorization built from scratch using NumPy

User & item bias terms included

Stochastic Gradient Descent (SGD) optimization

Train/Validation RMSE tracking

Hyperparameter tuning notebook (grid search)

PCA & t-SNE visualization of latent factors

Streamlit web app for interactive movie recommendations

Clean modular code in src/ for easy extension

📝 Resume-Ready Summary (Copy/Paste)

One-liner:
Built a Matrix Factorization movie recommender from scratch (NumPy) on MovieLens-100K with SGD, hyperparameter tuning, PCA/t-SNE visualization, and a Streamlit demo.

Two-liner:
Implemented a complete Matrix Factorization recommender using NumPy with SGD optimization, user/item biases, and RMSE-based validation. Developed hyperparameter tuning pipelines, latent-factor visualizations, and an interactive Streamlit interface with modular code and well-structured notebooks.

📚 Table of Contents

Quick Demo / Screenshot

Key Features

Installation

Usage

Notebooks

Project Structure

Implementation Notes

Presentation Tips

Contributing

Contact

🔧 Installation
Requirements

Python 3.8+

Recommended: virtual environment

Setup
# Clone repository
git clone https://github.com/punith624/movie-recommender.git
cd movie-recommender

# Create virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

▶️ Usage
1️⃣ Train the model
python main.py --train --k 30 --lr 0.007 --reg 0.02 --epochs 30


Model artifacts will be saved in /models/.

2️⃣ Launch the Streamlit demo
streamlit run app.py


Open your browser at:
👉 http://localhost:8501

3️⃣ Run notebooks

Open the .ipynb files inside the notebooks/ folder using Jupyter or VSCode.

📘 Notebooks
Notebook	Description
01_professional_notebook.ipynb	Full walkthrough with math, EDA, MF training, and evaluation
02_hyperparam_tuning.ipynb	Grid search for k, learning rate, regularization
03_latent_viz.ipynb	PCA & t-SNE visualization of latent factors
🗂 Project Structure
movie-recommender/
├── assets/                   # screenshots & visuals
│   └── demo.png
├── data/                     # dataset (ignored)
├── models/                   # trained models (ignored)
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


data/ and models/ are intentionally ignored by Git to keep the repo lightweight.

🧠 Implementation Notes

Prediction model:

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


Optimization:
SGD over rating triplets with L2 regularization.

Metrics:
RMSE on both train and validation sets.

Visualization:
PCA → global structure
t-SNE → fine similarity clusters


	​


Optimization:
SGD over rating triplets with L2 regularization.

Metrics:
RMSE on both train and validation sets.

Visualization:
PCA → global structure
t-SNE → fine similarity clusters

🎨 Presentation Tips (Interviews / Portfolio)

Show RMSE training curves (model quality)

Explain biases + MF formula briefly

Present Top-N recommendations for a user

Show PCA/t-SNE plots colored by genre

Mention runtime efficiency and scalability

Highlight clean modular code structure

📄 .gitignore (Recommended)
venv/
__pycache__/
*.pyc
data/
models/
assets/
.env

🤝 Contributing

Suggestions and pull requests are welcome!

📬 Contact

Punith Kumar
GitHub: https://github.com/punith624

Email:kumarpunith6864@gmail.com


