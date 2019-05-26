
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
# Gradient Descent
# Loss fuction J = SUM(r_actual - r_predict)**2 + normalization
# r_predict[a, m] = W[a].dot(U[m]) + b[a] + c[m] + mu
# b, c : bias terms  /  mu : global average

# Create the get_loss funtion
def get_loss(x):
    '''
    Calculating the loss
    input: the data in the form of "(user, movie) : rating"
    output: the mean_squared_error
    '''
    n = len(x)
    loss = 0
    for (a, m), r in x.items():
        pred = W[a].dot(U[m]) + b[a] + c[m] + mu
        loss += (r - pred)**2
    return loss/n

# Initialize parameters to update
k = 10                              # latent features
W = np.random.randn(N, k)
U = np.random.randn(M, k)
b = np.zeros(N)
c = np.zeros(M)
mu = np.mean(list(um_to_rating_te.values()))


# Train the model
epochs = 5
reg = 20                            # regularization penalty

tr_losses = []
te_losses = []

for epoch in range(epochs):

    # Gradient descent on W and b (the parameters related to the user a)
    for a in range(N):

        # Initialize variables for calculation
        term_1 = np.eye(k)*reg
        term_2 = np.zeros(k)
        b_a = 0

        # Updating the parameters along the movie ids while the user id is constant
        for m in user_to_movie[a]:
            r_a_m = um_to_rating_tr[(a, m)]

            term_1 += np.dot(U[m], U[m].T)
            term_2 += (r_a_m - b[a] - c[m] - mu)*U[m]
            b_a += r_a_m - U[m].dot(W[a]) - c[m] - mu

        W[a] = np.linalg.solve(term_1, term_2)
        b[a] = 1/(len(user_to_movie[a])*(1 + reg)) * b_a


    # Gradient descent on U and c (the parameters related to the movie m)
    for m in range(M):

        # Initialize variables for calculation
        term_1 = np.eye(k)*reg
        term_2 = np.zeros(k)
        c_m = 0

        # Updating the parameters along the user ids while the movie id is constant
        try:
            for a in movie_to_user[m]:
                r_a_m = um_to_rating_tr[(a, m)]

                term_1 += np.dot(W[a], W[a].T)
                term_2 += (r_a_m - b[a] - c[m] - mu)*W[a]
                c_m += r_a_m - W[a].dot(U[m]) - b[a] - mu

            U[b] = np.linalg.solve(term_1, term_2)
            c[m] = 1/(len(movie_to_user[m])*(1+reg)) * c_m

        # for the case the movie m has no rating
        except:
            pass

    tr_loss = get_loss(um_to_rating_tr)
    te_loss = get_loss(um_to_rating_te)
    print("{}th epoch train loss : {} , test lss : {}".format(epoch, tr_loss, te_loss))

    tr_losses.append(tr_loss)
    te_losses.append(te_loss)


print("total train loss: ", tr_losses)
print("Total test loss: ", te_losses)
