
import pandas as pd
import numpy as np
import pickle


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
# Redesigning the user_to_movie and movie_to_user dictionary to contain the rating data
# Create the dictionary in the form of "user_id : (movie_ids, rating_arr)"
# This makes the computaion speed faster using the power of numpy array
user_to_mr = {}
for a, movies in user_to_movie.items():
    rating_list = [um_to_rating_tr[(a, m)] for m in movies]
    rating_arr = np.array(rating_list)
    user_to_mr[a] = (movies, rating_arr)

# Create the dictionary in the form of " movie_id : (user_ids, rating_arr)"
movie_to_ur_tr = {}
for m, users in movie_to_user.items():
    rating_list = [um_to_rating_tr[(a, m)] for a in users]
    rating_arr = np.array(rating_list)
    movie_to_ur_tr[m] = (users, rating_arr)

# Repeat the same process with the test set
movie_to_ur_te = {}
for (a, m), r in um_to_rating_te.items():
    # If a movie appear for the first time, save the user and rating data as a list
    if m not in movie_to_ur_te:
        movie_to_ur_te[m] = [[a], [r]]
    # If a movie already exist, add the new user id and rating data to the corresponding index
    # Note that the final values will be "movie id : ( a list of user ids , a list of ratings )"
    else:
        movie_to_ur_te[m][0].append(a)
        movie_to_ur_te[m][1].append(r)

# Convert the list of ratings as an array for computation speed later on
for m, (users, ratings) in movie_to_ur_te.items():
    movie_to_ur_te[m][1] = np.array(ratings)



#####################################################
# Gradient Descent
# Loss fuction J = SUM(r_actual - r_predict)**2 + normalization
# r_predict[a, m] = W[a].dot(U[m]) + b[a] + c[m] + mu
# b, c : bias terms  /  mu : global average

# Create the get_loss funtion
def get_loss(x):
    '''
    Calculating the loss
    input: the data in the form of " movie : ( a list of user ids, an array of ratings )"
    output: the mean_squared_error
    '''
    n = 0
    loss = 0

    for m, (users, ratings) in x.items():
        pred = W[users].dot(U[m]) + b[users] + c[m] + mu
        loss += (ratings - pred).dot(ratings - pred)

        # Count the total number of ratings
        n += len(ratings)

    return loss/n


# Initialize parameters to update
k = 10                              # latent features
W = np.random.randn(N, k)
U = np.random.randn(M, k)
b = np.zeros(N)
c = np.zeros(M)
mu = np.mean(list(um_to_rating_te.values()))


# Train the model
epochs = 10
reg = 0.1                # regularization penalty

tr_losses = []
te_losses = []

for epoch in range(epochs):

    # Gradient descent on W and b (the parameters related to the user a)
    for a in range(N):

        # Now we can speed up the computation with the power of numpy
        ###########################################
        movies, rating_arr = user_to_mr[a]
        term_1 = np.eye(k)*reg + np.dot(U[movies].T, U[movies])
        term_2 = (rating_arr - b[a] - c[movies] - mu).dot(U[movies])
        b_a = (rating_arr - U[movies].dot(W[a]) - c[movies] - mu).sum()
        ###########################################

        W[a] = np.linalg.solve(term_1, term_2)
        b[a] = 1/(len(user_to_movie[a])*(1 + reg)) * b_a

    # Gradient descent on U and c (the parameters related to the movie m)
    for m in range(M):
        try:
            ###########################################
            users, rating_arr = movie_to_ur[m]
            term_1 = np.eye(k)*reg + np.dot(W[users].T, W[users])
            term_2 = (rating_arr - b[users] - b[m] - mu).dot(W[users])
            c_m = (rating_arr - W[users].dot(U[m]) - b[users] - mu).sum()
            ###########################################

            U[b] = np.linalg.solve(term_1, term_2)
            c[m] = 1/(len(movie_to_user[m])*(1+reg)) * c_m

        # for the case the movie m has no rating
        except:
            pass

    tr_loss = get_loss(movie_to_ur_tr)
    te_loss = get_loss(movie_to_ur_te)
    print("{}th epoch train loss : {} , test lss : {}".format(epoch, tr_loss, te_loss))

    tr_losses.append(tr_loss)
    te_losses.append(te_loss)


print("total train loss: ", tr_losses)
print("Total test loss: ", te_losses)
