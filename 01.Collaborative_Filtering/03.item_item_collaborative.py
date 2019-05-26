
import pandas as pd
import numpy as np
import pickle

from sortedcontainers import SortedList

dir = '../'

# Load the data
with open(dir + 'data/user_to_movie.json', 'rb') as f:
    user_to_movie = pickle.load(f)

with open(dir + 'data/movie_to_user.json', 'rb') as f:
    movie_to_user = pickle.load(f)

with open(dir + 'data/um_to_rating_tr.json', 'rb') as f:
    um_to_rating_tr = pickle.load(f)

with open(dir + 'data/um_to_rating_te.json', 'rb') as f:
    um_to_rating_te = pickle.load(f)


# Count the number of users
N = np.max(list(user_to_movie.keys())) + 1                               # User id starts from 0

# Count the number of movies
m_tr = np.max(list(movie_to_user.keys()))

# Get the maximum movie id from the test set for the movies not in the train set
te_movie_list = [m for (u, m), r in um_to_rating_te.items()]
m_te = np.max(te_movie_list)

M = max(m_tr, m_te) + 1
print("The size of the data: {} users & {} movies".format(N, M))


#####################################################
# Get the average and deviation of ratings for each user and the similarities between users
# 1. The average rating for each movie : r_avg                           ----> the list of averages
# 2. The deviation of rating for each user and movie : r_dev_dic_a       ----> the list of deviations
# 3. The similarities between movie a and b : w_mn                       ----> the list of neighbors with the weights

K = 20                  # the number of neights to consider
similarity_limit = 5    # the minimum number of movies users must have in common

averages = []           # ----> the list of averages
deviations = []         # ----> the list of deviations
neighbors = []          # ----> the list of neighbors with the weights

for m in range(M):

    # Get the user lists who rated movie m
    users_m = movie_to_user[m]

    # Create the set of the users to get the common user list with movie n
    users_m_set = set(users_m)

    # Create the "user : rating" dictionary and get the rating average to the movie m
    r_dic_m = { a : um_to_rating_tr[(a, m)] for a in users_m }
    r_avg_m = np.mean(list(r_dic_m.values()))                                   # average rating of the movie m

    # Create the "user : rating deviation" dictionary and get the deviation to the movie m
    r_dev_dic_m = { a : (r - r_avg_m) for a, r in r_dic_m.items() }             # how the user ratings deviate about the movie m
    r_dev_arr_m = np.array(list(r_dev_dic_m.values()))
    r_sigma_m = np.sqrt(np.dot(r_dev_arr_m, r_dev_arr_m))                       # for calculating the movie similarities

    # Save the average and deviation value
    averages.append(r_avg_m)
    deviations.append(r_dev_arr_m)

    w_neighbor = SortedList()
    for n in range(M):
        if n != m:

            # Get the user lists rated by movie n
            users_n = movie_to_user[n]

            # Create the set of the users to get the common user list with movie n
            users_n_set = set(users_n)
            users_m_n = (users_m_set & users_n_set)                             # common users by movie m & n (Intersection)

            # Get the rating average and deviation of the other movies whose similarities are above the limit
            if len(users_m_n) > similarity_limit:

                # Create the "user : rating" dictionary and get the rating average by movie n
                r_dic_n = { a : um_to_rating_tr[(a, n)] for a in users_n }
                r_avg_n = np.mean(list(r_dic_n.values()))                       # average rating to the movie n

                # Create the "user : rating deviation" dictionary and get the deviation to the movie n
                r_dev_dic_n = { a : (r - r_avg_n) for a, r in r_dic_n.items() }
                r_dev_arr_n = np.array(list(r_dev_dic_n.values()))
                r_sigma_n = np.sqrt(np.dot(r_dev_arr_n, r_dev_arr_n))           # for calculating the movie similarities

                # movie similarities (Pearson correlation)
                numerator = sum(r_dev_dic_m[a] * r_dev_dic_n[a] for a in users_m_n)
                w_mn = numerator / (r_sigma_m * r_sigma_n)

                # Add to the sorted list (negative weights for descending)
                w_neighbor.add((-w_mn, n))

                # If the number of neighbors is obove the limit K, delete the last one
                if len(w_neighbor) > K:
                    del w_neighbor[-1]

    # Save the neightbor
    neighbors.append(w_neighbor)

    if a % 1000 == 0:
        print("==========Processed: {}%".format(a//N))            # tracking the preprocessing


#####################################################
# averages : THe list of the average ratings to each movie
# deviations :  The list of the r_dev_arr to each movie

def predict_scoring(a, m):
    '''
    Predicting the rating of user a to movie b
    input : user_id (a) and movie_id (m)
    output : the predicted rating
    '''
    # Initialization
    numerator = 0
    denominator = 0

    for w_mn, n in neighbors[m]:
        try:
            w_mn *= -1
            numerator += w_mn * deviations[n][a]
            denominator += abs(w_mn)

        # For the case that the neighbors of the given movie n hasn't rate the given user a
        except:
            pass

    # If the sum of weights is 0 (the sum of the similarities values = 0)
    if denominator == 0:
        pred = averages[m]
    else:
        pred = averages[m] + numerator / denominator

    # The fixed range of rating is between .5 and 5
    pred = min(5, pred)
    pred = max(.5, pred)

    return pred


def get_loss(pred, actual):
    '''
    Calculating the loss by MSE
    input: the list of the prediction and actual values
    output : mean_squared_error
    '''
    loss = np.array(pred) - np.array(actual)
    return np.mean(loss**2)


# Get the prediction and actual value from train set
tr_preds = []
tr_actual = []

for (a, m), r in um_to_rating_tr.items():
    # Predict the score of movie m by user a
    pred = predict_scoring(a, m)

    tr_preds.append(pred)
    tr_actual.append(r)

# Get the prediction and actual value from test set
te_preds = []
te_actual = []

for (a, m), r in um_to_rating_te.items():
    # Predict the score of movie m by user a
    pred = predict_scoring(a, m)

    te_preds.append(pred)
    te_actual.append(r)


# Get the loss
tr_loss = get_loss(tr_preds, tr_actual)
tr_loss = get_loss(te_preds, te_actual)

print("The loss of train set : %.4f" % tr_loss)
print("The loss of test set :  %.4f" % te_loss)
