import pandas as pd

url = "https://raw.githubusercontent.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/master/IMDB%20Dataset.csv"
df = pd.read_csv(url)
df.to_csv("IMDB Dataset.csv", index=False)
print("Dataset downloade
d!")
