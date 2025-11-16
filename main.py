import pandas as pd
import os
import zipfile

# Make sure data folder exists
os.makedirs("data", exist_ok=True)

print("Downloading MovieLens 100k...")
import urllib.request
urllib.request.urlretrieve(
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-100k.zip"
)

print("Extracting...")
with zipfile.ZipFile("ml-100k.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# Convert u.item → movies.csv
movies_out = "data/movies.csv"
with open("ml-100k/u.item", encoding="latin-1") as f:
    lines = f.readlines()

with open(movies_out, "w", encoding="utf-8") as f:
    f.write("movieId,title,genres\n")
    for line in lines:
        parts = line.strip().split("|")
        movieId = parts[0]
        title = parts[1].replace('"', '')
        genres = "|".join(parts[5:]).replace('"', '')
        f.write(f'{movieId},"{title}","{genres}"\n')

# Convert u.data → ratings.csv
ratings_out = "data/ratings.csv"
with open("ml-100k/u.data") as f_in, open(ratings_out, "w") as f_out:
    f_out.write("userId,movieId,rating,timestamp\n")
    for line in f_in:
        u, i, r, t = line.split()
        f_out.write(f"{u},{i},{r},{t}\n")

print("DONE — ratings.csv and movies.csv created!")


   