
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
# 1. The average rating for each user : r_avg                           ----> the list of averages
# 2. The deviation of rating for each user and movie : r_dev_dic_a      ----> the list of deviations
# 3. The similarities between user a and b : w_ab                       ----> the list of neighbors with the weights

K = 25                  # the number of neights to consider
similarity_limit = 5    # the minimum number of movies users must have in common

averages = []           # ----> the list of averages
deviations = []         # ----> the list of deviations
neighbors = []          # ----> the list of neighbors with the weights

for a in range(N):

    # Get the movie lists rated by user a
    movies_a = user_to_movie[a]

    # Create the set of the movies to get the common movie list with user b
    movies_a_set = set(movies_a)

    # Create the "movie : rating" dictionary and get the rating average by user a
    r_dic_a = { m : um_to_rating_tr[(a, m)] for m in movies_a }
    r_avg_a = np.mean(list(r_dic_a.values()))                           # average rating of user a

    # Create the "movie : rating deviation" dictionary and get the deviation by user a
    r_dev_dic_a = { m : (r - r_avg_a) for m, r in r_dic_a.items() }
    r_dev_arr_a = np.array(list(r_dev_dic_a.values()))
    r_sigma_a = np.sqrt(np.dot(r_dev_arr_a, r_dev_arr_a))                # for calculating the user similarities

    # Save the average and deviation value
    averages.append(r_avg_a)
    deviations.append(r_dev_arr_a)

    w_neighbor = SortedList()
    for b in range(N):
        if b != a:

            # Get the movie lists rated by user b
            movies_b = user_to_movie[b]

            # Create the set of the movies to get the common movie list with user b
            movies_b_set = set(movies_b)
            movies_a_b = (movies_a_set & movies_b_set)                   # common movies by user a & b (Intersection)

            # Get the rating average and deviation by the other users whose similarities are above the limit
            if len(movies_a_b) > similarity_limit:

                # Create the "movie : rating" dictionary and get the rating average by user b
                r_dic_b = { m : um_to_rating_tr[(b, m)] for m in movies_b }
                r_avg_b = np.mean(list(r_dic_b.values()))                    # average rating of user b

                # Create the "movie : rating deviation" dictionary and get the deviation by user b
                r_dev_dic_b = { m : (r - r_avg_b) for m, r in r_dic_b.items() }
                r_dev_arr_b = np.array(list(r_dev_dic_b.values()))
                r_sigma_b = np.sqrt(np.dot(r_dev_arr_b, r_dev_arr_b))    # for calculating the user similarities

                # User similarities (Pearson correlation)
                numerator = sum(r_dev_dic_a[m] * r_dev_dic_b[m] for m in movies_a_b)
                w_ab = numerator / (r_sigma_a * r_sigma_b)

                # Add to the sorted list (negative weights for descending)
                w_neighbor.add((-w_ab, b))

                # If the number of neighbors is obove the limit K, delete the last one
                if len(w_neighbor) > K:
                    del w_neighbor[-1]

    # Save the neightbor
    neighbors.append(w_neighbor)

    if a % 1000 == 0:
        print("==========Processed: {}%".format(a//N))            # tracking the preprocessing


#####################################################
# averages : THe list of the average ratings for each user
# deviations :  The list of the r_dev_arr for each user

def predict_scoring(a, m):
    '''
    Predicting the rating of user a to movie b
    input : user_id (a) and movie_id (m)
    output : the predicted rating
    '''
    # Initialization
    numerator = 0
    denominator = 0
    for w_ab, b in neighbors[a]:
        try:
            w_ab *= -1
            numerator += w_ab * deviations[b][m]
            denominator += abs(w_ab)

        # For the case that the neighbors of user a didn't rate the given movie m
        except:
            pass

    # If the sum of weights is 0 (the sum of the similarities values = 0)
    if denominator == 0:
        pred = averages[a]
    else:
        pred = averages[a] + numerator / denominator

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
