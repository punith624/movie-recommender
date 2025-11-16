import os
import urllib.request
import zipfile

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

print("Downloading MovieLens 100k...")
urllib.request.urlretrieve(
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-100k.zip"
)

print("Extracting...")
with zipfile.ZipFile("ml-100k.zip", "r") as zip_ref:
    zip_ref.extractall(".")

print("Creating movies.csv...")
with open("ml-100k/u.item", encoding="latin-1") as f_in, open("data/movies.csv", "w", encoding="utf-8") as f_out:
    f_out.write("movieId,title,genres\n")
    for line in f_in:
        parts = line.strip().split("|")
        movieId = parts[0]
        title = parts[1].replace('"', "")
        genres = "|".join(parts[5:]).replace('"', "")
        f_out.write(f'{movieId},"{title}","{genres}"\n')

print("Creating ratings.csv...")
with open("ml-100k/u.data") as f_in, open("data/ratings.csv", "w") as f_out:
    f_out.write("userId,movieId,rating,timestamp\n")
    for line in f_in:
        u, i, r, t = line.split()
        f_out.write(f"{u},{i},{r},{t}\n")

print("DONE — movies.csv and ratings.csv created successfully!")
