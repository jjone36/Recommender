import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import SGD

dir = '../'

# Load the files
A = laod_npz(dir + 'data/tr.npz')
A_te = laod_npz(dir + 'data/te.npz')


# Create masks for tracking non-missing values
mask = (A>0)*1.
mask_te = (A_te>0)*1.


# Copy the original data before shuffling
A_copy = A.copy()
A_te_copy = A_te.copy()

mask_copy = mask.copy()
mask_te_copy = mask_te.copy()

N, M = A.shape
mu = A.sum() / mask.sum()
print("The size of the data: {} , {}".format(N, M))
print("The mean value is: ", mu)


epochs = 10
batch_size = 128
reg = .00001


# Modeling
i = Input(shape = (M, ))
x = Dropout(.7)(i)
x = Dense(700, activation= 'tanh', kernel_regularizer=l2(reg))(x)
#x = Dropout(.5)(x)
x = Dense(M, kernel_regularizer= l2(reg))(x)


def custom_loss(y_true, y_pred):
  mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
  diff = y_pred - y_true
  sqdiff = diff * diff * mask
  sse = K.sum(K.sum(sqdiff))
  n = K.sum(K.sum(mask))
  return sse / n


def generator(A, M):
  while True:
    A, M = shuffle(A, M)
    for i in range(A.shape[0] // batch_size + 1):
      upper = min((i+1)*batch_size, A.shape[0])
      a = A[i*batch_size:upper].toarray()
      m = M[i*batch_size:upper].toarray()
      a = a - mu * m # must keep zeros at zero!
      # m2 = (np.random.random(a.shape) > 0.5)
      # noisy = a * m2
      noisy = a # no noise
      yield noisy, a


def test_generator(A, M, A_test, M_test):
  # assumes A and A_test are in corresponding order
  # both of size N x M
  while True:
    for i in range(A.shape[0] // batch_size + 1):
      upper = min((i+1)*batch_size, A.shape[0])
      a = A[i*batch_size:upper].toarray()
      m = M[i*batch_size:upper].toarray()
      at = A_test[i*batch_size:upper].toarray()
      mt = M_test[i*batch_size:upper].toarray()
      a = a - mu * m
      at = at - mu * mt
      yield a, at


model = Model(i, x)
model.compile(loss=custom_loss,
              optimizer=SGD(lr=0.08, momentum=0.9),
              metrics=[custom_loss])

r = model.fit_generator(generator(A, mask),
                        validation_data=test_generator(A_copy, mask_copy, A_test_copy, mask_test_copy),
                        epochs=epochs,
                        steps_per_epoch=A.shape[0] // batch_size + 1,
                        validation_steps=A_test.shape[0] // batch_size + 1)
print(r.history.keys())
