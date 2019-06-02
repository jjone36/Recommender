
import pandas as pd
import numpy as np

dir = '../'

# Load the data
rt = pd.read_csv(dir + 'data/rating.csv')

# Make the user Id starts from 0
df.userId -= 1

# create a mapping for movie ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
  movie2idx[movie_id] = count
  count += 1

# add them to the data frame
df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

# Save the file
df.to_csv(dir + 'data/rating_2.csv', index = False)
