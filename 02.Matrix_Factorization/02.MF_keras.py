
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.utils import shuffle
from sortedcontainers import SortedList

from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.models import Model

# 1. Loading the data
df = pd.read_csv('data/rating.csv')
mo = pd.read_csv('data/movie.csv')

# Joing the two data frame
df2 = pd.merge(df, mo, how = 'inner', on = ['movieId'])
df2 = df2.drop(columns = ['timestamp', 'genres'])

# Make the user Id starts from 0
df2.userId -= 1


# 2. Preprocessing 
# Create a mapping for movie ids
movie_set = set(df.movieId.values)
movie_idx = {}
i = 0

for k in movie_set:
    movie_idx[k] = i
    i += 1

df['movie_idx'] = df.apply(lambda x: movie_idx[x.movieId], axis = 1)

N = df.userId.max() + 1
M = df.movieId.max() + 1

print("The number of Users is ", N)
print("The number of Movies is ", M)

# Splitting the data
cut = int(0.8*len(df))

df = shuffle(df)
tr = df.iloc[:cut]
te = df.iloc[cut:]

rating_avg = tr.rating.mean()    # global average

X_tr = [tr.userId.values, tr.movieId.values]
y_tr = tr.rating.values - rating_avg

X_te = [te.userId.values, te.movieId.values]
y_te = te.rating.values - rating_avg

# 3. Modeling
K = 10                           # Latent Dimensionality
reg = 0                          # regularity penalty

### 3-1. baseline
# Moedling
epochs = 10

# Input layer
u = Input(shape = (1, ))
m = Input(shape = (1, ))

# Embedding layer
u_embedding = Embedding(N, K, embeddings_regularizer= l2(reg))(u)
m_embedding = Embedding(M, K, embeddings_regularizer= l2(reg))(m)
x = Dot(axes = 2)([u_embedding, m_embedding])

u_bias = Embedding(N, 1, embeddings_regularizer= l2(reg))(u)
m_bias = Embedding(M, 1, embeddings_regularizer= l2(reg))(m)

x = Add()([x, u_bias, m_bias])
x = Flatten()(x)

### 3-2. 
# Input layer
u = Input(shape = (1, ))
m = Input(shape = (1, ))

# Embedding layer
u_embedding = Embedding(N, K)(u)
m_embedding = Embedding(M, K)(m)

u_embedding = Flatten()(u_embedding)
m_embedding = Flatten()(m_embedding)

x = Concatenate()([u_embedding, m_embedding])

# the neural network
x = Dense(400)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)

# x = Dropout(0.5)(x)
# x = Dense(100)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)

x = Dense(1)(x)

# building and optimizatiion
model = Model(inputs = [u, m], outputs = x)
model.compile(optimizer = SGD(lr = .08, momentum = .9), 
              loss = 'mse', 
              metrics = ['mse'])

model.summary()

### 3-3. Residual model
# input layer
u = Input(shape = (1, ))
m = Input(shape = (1, ))

u_embedding = Embedding(N, K)(u)
m_embedding = Embedding(M, K)(m)

# main branch
u_bias = Embedding(N, 1)(u)
m_bias = Embedding(M, 1)(m)

x = Dot(axes = 2)([u_embedding, m_embedding])
x = Add()([x, u_bias, m_bias])
x = Flatten()(x)

# side brance
u_embedding = Flatten()(u_embedding)
m_embedding = Flatten()(m_embedding)
x2 = Concatenate()([u_embedding, m_embedding])
x2 = Dense(400)(x2)
x2 = Activation('elu')(x2)
# x2 = Dropout(.5)(x2)
x2 = Dense(1)(x2)

# Combine two brances
X = Add()([x, x2])


### 3-3. Autoencoder




# 4. Evaluation 
# building and optimizatiion
model = Model(inputs = [u, m], outputs = x)
model.compile(loss = 'mse', optimizer = SGD(lr = .08, momentum = .9), 
             metrics = ['mse'])

model.summary()

# Fitting the model 
history = model.fit(x = X_tr, y = y_tr,
                    epochs = epochs, 
                    batch_size = 128,
                    validation_data = (X_te, y_te))

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