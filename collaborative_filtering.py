
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.utils import shuffle
from sortedcontainers import SortedList


df = pd.read_csv('data/rating.csv')
mo = pd.read_csv('data/movie.csv')

# Joing the two data frame
df2 = pd.merge(df, mo, how = 'inner', on = ['movieId'])
df2 = df2.drop(columns = ['timestamp', 'movieId', 'genres'])

# Make the user Id starts from 0
df2.userId -= 1

N = df2.userId.max() + 1
M = df2.title.nunique() + 1
print("The number of Users is ", N)
print("The number of Movies is ", M)

user_ids_count = Counter(df2.userId)
movie_ids_count = Counter(df2.title)

# Filter the data
df_sub = df2[df2.userId.isin(user_ids) & df2.title.isin(movie_ids)]

# Indexing the user and movie list
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
df_sub['movie_idx'] = df_sub.title.apply(lambda x: movie_dic[x])
df_sub = df_sub.reset_index(drop = True)


cut = int(0.8*len(df_sub))

df_sub = shuffle(df_sub)
tr = df_sub.iloc[:cut]
te = df_sub.iloc[cut:]

tr = tr.reset_index(drop = True)
te = te.reset_index(drop = True)

print("The size of train : ", len(tr))
print("The size of test : ", len(te))


user_to_movie = {}
movie_to_user = {}
um_to_rating = {}

def making_dic(x):

    a = int(x.user_idx)
    m = int(x.movie_idx)
    r = x.rating

    # make a dictionary for "user to movie"
    if a not in user_to_movie:
        user_to_movie[a] = [m]
    else:
        user_to_movie[a].append(m)

    # make a dictionary for "movie to user"
    if m not in movie_to_user:
        movie_to_user[m] = [a]
    else:
        movie_to_user[m].append(a)

    # make rating dictionary
    um_to_rating[(a, m)] = r

    um_to_rating_te = {}

def making_dic_te(x):
    
    a = int(x.user_idx)
    m = int(x.movie_idx)
    r = x.rating

    um_to_rating_te[(a, m)] = r

te.apply(making_dic_te, axis = 1)


with open('user2movie.json', 'wb') as f:
  pickle.dump(user2movie, f)

with open('movie2user.json', 'wb') as f:
  pickle.dump(movie2user, f)

with open('usermovie2rating.json', 'wb') as f:
  pickle.dump(usermovie2rating, f)

with open('usermovie2rating_test.json', 'wb') as f:
  pickle.dump(usermovie2rating_test, f)
