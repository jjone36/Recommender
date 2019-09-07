import findspark
findspark.init('/home/jjone/spark-2.3.1-bin-hadoop2.7')

from pyspark.mllib.recomendation import ALS, MatrixFactorizationModel, Rating

dir = '../'
# Load the data
data = pd.read_csv(dir + 'data/rating_2.csv')

# Filter out header
header = data.first()
dta = df.filter(lambda row: row != header)

# Convert into a sequence of Rating objects
ratings = data.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Split into train and test
train, test = ratings.randomSplit([0.8, 0.2])

# Train the model
K = 10
epochs = 10
model = ALS.train(train, K, epochs)

# Evaluate the model

# train
x = train.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(p)
# joins on first item: (user_id, movie_id)
# each row of result is: ((user_id, movie_id), (rating, prediction))
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("train mse: %s" % mse)


# test
x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("test mse: %s" % mse)
