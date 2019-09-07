import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
import sklearn.utils import shuffle
import tensorflow as tf


def oh_encoder(X, K):
    '''
    making one-hot-encoding the 2-D input data to 3 dimensionals
    input: (N, D) matrix whose values are the float rating
    output: (N, D, K) matrix whose values are the encoded values of the input data
    '''
    # input
    N, D = X.shape
    # output
    Y = np.zeros((N, D, K))
    # convert float ratings to integer
    for n, d in zip(*X.nonzero()):
        k = int(X[n, d]*2 - 1)
        Y[n, d, k] = 1
    return Y


def oh_mask(X, K):
    '''
    making zero values into 0 and non-zero values into 1
    input: (N, D) matrix whose values are the float rating
    output: (N, D, K) matrix whose values are binary
    '''
    # input
    N, D = X.shape
    # output
    Y = np.zeros((N, D, K))
    # convert zero to 0, non-zero to 1
    for n, d in zip(*X.nonzero()):
        Y[n, d, :] = 1
    return Y


one_to_ten = np.arange(10) + 1   # values from 1 to 10
def convert_probs_to_ratings(pred):
    '''
    converting probablitic predictions to int
    input: (N, D, K) matrix with probablitic predictions
    output: integer predicted ratings ranging from 1 to 5
    '''
    return probs.dot(one_to_ten) / 2


# customed dot functions
def dot_1(V, W):
    '''
    input- V: k batch of visible units with the shape of (N, D, K)
         - W: parameters with the shape of (D, K, M)
    output: the output with the shape of (N, M)
    '''
    return tf.tensordot(V, W, axes = [[1, 2], [0, 1]])

def dot_2(H, W):
    '''
    input- V: the hidden units with the shape of (N, M)
         - W: parameters with the shape of (D, K, M)
    output: the output with the shape of (N, D, K)
    '''
    return tf.tensordot(H, W, axes = [[1], [2]])


# Create the RBM class
class RBM(object):
    def __init__(self, D, M, K):
        self.D = D  # input size
        self.M = M  # hidden units
        self.K = K  # number of ratings
        self.build(D, M, K)

    def build(self, D, M, K):
        # params
        self.W = tf.Variable(tf.random_normal(shape=(D, K, M)) * np.sqrt(2.0 / M))
        self.c = tf.Variable(np.zeros(M).astype(np.float32))
        self.b = tf.Variable(np.zeros((D, K)).astype(np.float32))

        # data
        self.X_in = tf.placeholder(tf.float32, shape=(None, D, K))
        self.mask = tf.placeholder(tf.float32, shape=(None, D, K))

        # conditional probabilities
        V = self.X_in
        p_h_given_v = tf.nn.sigmoid(dot1(V, self.W) + self.c)
        self.p_h_given_v = p_h_given_v # save for later

        # draw a sample from p(h | v)
        r = tf.random_uniform(shape=tf.shape(p_h_given_v))
        H = tf.to_float(r < p_h_given_v)

        # draw a sample from p(v | h)
        logits = dot2(H, self.W) + self.b
        cdist = tf.distributions.Categorical(logits=logits)
        X_sample = cdist.sample()
        X_sample = tf.one_hot(X_sample, depth=K)
        X_sample = X_sample * self.mask

        # objective
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(objective)

        # build the cost
        # we won't use this to optimize the model parameters
        # just to observe what happens during training
        logits = self.forward_logits(self.X_in)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.X_in, logits=logits))

        # to get the output
        self.output_visible = self.forward_output(self.X_in)

        initop = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(initop)



# load the files and fit the rbm model
def main():
    A = laod_npz('tr.npz')
    A_te = load_npz("te.npz")
    mask = (A > 0)*1.
    mask_te = (A_te > 0)*1.

    N, M = A.shape
    rbm = RBM(M, 50, 10)
    rbm.fit(A, mask, A_te, mask_te)

# execution
if __name__ == '__main__':
    main()
