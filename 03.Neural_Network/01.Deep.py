
import pandas as pd
import numpy as np
import matplotlib.pyplot import plt
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

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

X_tr = tr[['userId', 'movie_idx']]
mu = tr.rating.mean()
y_tr = tr.rating - mu

X_te = te[['userId', 'movie_idx']]
y_te = te.rating - mu

# Modeling
K = 10                  # latent Dimensionality
reg = 0.                # regularization penalty
epochs = 10

# Input layer
u = Input(shape = (1, ))
m = Input(shape = (1, ))

# Embedding Layer
u_embedding = Embedding(input_dim= N, output_dim= K, embeddings_regularizer=l2(reg))(u)  # (N, 1, K)
m_embedding = Embedding(input_dim= M, output_dim= K, embeddings_regularizer=l2(reg))(m)  # (M, 1, K)


#####################################################
# Connect to the deeper layer
u_embedding = Flatten()(u_embedding)          # (N, K)
m_embedding = Flatten()(m_embedding)          # (N, K)
x = Concatenate()(u_embedding, m_embedding)   # (N, 2K)

# the neural network
x = Dense(400)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(100)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
x = Dense(1)(x)
#####################################################


model = Model(inputs = [u, m], outputs = x)
model.compile(optimizer= SGD(lr = .01, momentum = .9),
              loss = 'mse',
              metrics = ['mse'])              # due to regularization terms

# Train the model
r = model.fit(X_tr, y_tr,
              epochs=epochs,
              batch_size=128,
              validation_data= [X_te, y_te])

# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()


# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()
