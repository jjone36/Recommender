
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.utils import shuffle


rt = pd.read_csv('data/rating.csv')
mo = pd.read_csv('data/movie.csv')

# Step 1. Joing the two data frame
df = pd.merge(rt, mo, how = 'inner', on = ['movieId'])
df = df.drop(columns = ['timestamp', 'title', 'genres'])


# Step 2. Make the user Id starts from 0
df.userId -= 1

N = df.userId.max() + 1
M = df.movieId.nunique() + 1

print("The number of Users : ", N)
print("The number of Movies : ", M)

user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movieId)


# Step 3. Choose the numbers to subset
n = 10000
m = 2000

user_ids = [col for col, idx in user_ids_count.most_common(n)]
movie_ids = [col for col, idx in movie_ids_count.most_common(m)]

# Filter the data only with the user id and movie id mostly rated
df_sub = df[df.userId.isin(user_ids) & df.movieId.isin(movie_ids)]


# Step 4. Create user id and movie id columns
user_dic = {}
i = 0
for k in user_ids:
    user_dic[k] = i
    i += 1

movie_dic = {}
i = 0
for k in movie_ids:
    movie_dic[k] = i
    i += 1

df_sub['user_idx'] = df_sub.userId.apply(lambda x: user_dic[x])
df_sub['movie_idx'] = df_sub.movieId.apply(lambda x: movie_dic[x])
df_sub = df_sub.reset_index(drop = True)


# Step 5. Split the data into train and test set
cut = int(0.8*len(df_sub))

df_sub = shuffle(df_sub)
tr = df_sub.iloc[:cut]
te = df_sub.iloc[cut:]

tr = tr.reset_index(drop = True)
te = te.reset_index(drop = True)

print("The size of train : ", len(tr))
print("The size of test : ", len(te))


# Step 6. Create user, movie look-up dictionary with train set
user_to_movie = {}
movie_to_user = {}
um_to_rating_tr = {}
count = 0

def making_user_movie_dics_train(row):

    global count

    a = int(row.user_idx)    # The user id of the given row
    m = int(row.movie_idx)   # Tue movie id of the given row
    r = row.rating           # The rating that the user gave to the movie

    # Fill the "user to movie" dict
    if a not in user_to_movie:
        user_to_movie[a] = [m]
    else:
        user_to_movie[a].append(m)

    # Fill the "movie to user" dict
    if m not in movie_to_user:
        movie_to_user[m] = [a]
    else:
        movie_to_user[m].append(a)

    # Fill the "(user, movie) to rating" dict
    um_to_rating_tr[(a, m)] = r

    count += 1
    if count % 100000 == 0:
        print("==========Processed: %.2f" % (count/len(tr)))

# Apply the function
_ = tr.apply(making_user_movie_dics_train, axis = 1)


# Step 7. Create user, movie look-up dictionary with test set
um_to_rating_te = {}
count = 0

def making_user_movie_dics_test(x):

    global count

    a = int(x.user_idx)
    m = int(x.movie_idx)
    r = x.rating

    # Fill the "(user, movie) to rating" dict
    um_to_rating_te[(a, m)] = r

    count += 1
    if count % 100000 == 0:
        print("==========Processed: %.2f" % (count/len(te)))

_ = te.apply(making_user_movie_dics_test, axis = 1)


# Step 8. Save the dictionary files
with open('user_to_movie.json', 'wb') as f:
  pickle.dump(user_to_movie, f)

with open('movie_to_user.json', 'wb') as f:
  pickle.dump(movie_to_user, f)

with open('um_to_rating_tr.json', 'wb') as f:
  pickle.dump(um_to_rating_tr, f)

with open('um_to_rating_te.json', 'wb') as f:
  pickle.dump(um_to_rating_te, f)
