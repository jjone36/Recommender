import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz


dir = '../'

# Load the data
rt = pd.read_csv(dir + 'data/rating_2.csv')

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

print("The number of Users : ", N)
print("The number of Movies : ", M)


# Splitting the data into train and test set
df = shuffle(df)

cut = int(0.8*len(df))
tr = df.iloc[:cut]
te = df.iloc[cut:]

A = lil_matrix((N, M))

def update_tr(row):
    a = int(row.userId)
    m = int(row.movie_idx)
    A[a, m] = row.rating

tr.apply(update_tr, axis = 1)

A = A.tocsc()
mask = (A > 0)
save_npz(dir + "data/tr.npz", A)


A_te = lil_matrix((N, M))

def update_te(row):
    a = int(row.userId)
    m = int(row.movie_idx)
    A_te[a, m] = row.rating

te.apply(update_te, axis = 1)

A_te = A_te.tocsc()
mask_te = (A_te > 0)
save_npz(dir + "data/te.zpz", A_te)
