import math

import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Read in the training and test sets
train_df = pd.read_csv("./iris_train.txt", sep = ",")
test_df = pd.read_csv("./iris_test.txt", sep = ",")

# Split the train data into groups seperated by label
train_setosa = train_df.loc[train_df["is_setosa"] == 1.0]
train_versicolor = train_df.loc[train_df["is_versicolor"] == 1.0]
train_virginica = train_df.loc[train_df["is_virginica"] == 1.0]

# Extract features and labels of training data set as NumPy arrays
sepal_lengths_train = np.array(train_df["sepal_length"])
sepal_widths_train = np.array(train_df["sepal_width"]) 
petal_lengths_train = np.array(train_df["petal_length"])
petal_widths_train = np.array(train_df["petal_width"]) 
is_setosa_train = np.array(train_df["is_setosa"])
is_versicolor_train = np.array(train_df["is_versicolor"])
is_virginica_train = np.array(train_df["is_virginica"]) 

# Extract features and labels of test data set as NumPy arrays
sepal_lengths_test = np.array(test_df["sepal_length"])
sepal_widths_test = np.array(test_df["sepal_width"]) 
petal_lengths_test = np.array(test_df["petal_length"])
petal_widths_test = np.array(test_df["petal_width"]) 
is_setosa_test = np.array(test_df["is_setosa"])
is_versicolor_test = np.array(test_df["is_versicolor"])
is_virginica_test = np.array(test_df["is_virginica"]) 

# Create placeholders to feed data into during training process
x1 = tf.compat.v1.placeholder("float") # sepal lengths
x2 = tf.compat.v1.placeholder("float") # sepal widths
x3 = tf.compat.v1.placeholder("float") # petal lengths
x4 = tf.compat.v1.placeholder("float") # petal widths
y = tf.compat.v1.placeholder("float")

# Declare trainable TensorFlow variables for the weights and bias and initialize them randomly
w1 = tf.Variable(0.0, name = "w1") # sepal length weight
w2 = tf.Variable(0.0, name = "w2") # sepal width weight
w3 = tf.Variable(0.0, name = "w3") # petal length weight
w4 = tf.Variable(0.0, name = "w4") # petal width weight
b = tf.Variable(0.0, name = "b") # bias

# Define hyperparameters of model
learning_rate_setosa = 0.1
learning_rate_versicolor = 0.1
learning_rate_virginica = 0.1
num_epochs = 50

# The linear layer of the model
z = (w1 * x1) + (w2 * x2) + (w3 * x3) + (w4 * x4) + b

# Logistic regression model using a sigmoid function
y_pred = tf.divide(1.0, tf.add(1.0, tf.math.exp(-z)))

# Takes in a float x and returns log_10 of x
def log_ten(x):
	num = tf.math.log(x)
	denom = tf.math.log(10.0)
	return num / denom

# Takes in experimental and predicted labels and returns a log loss function
def create_log_loss(y, y_pred):
	part1 = tf.multiply(-y, log_ten(y_pred))
	part2 = tf.multiply(tf.subtract(1.0, y), log_ten(tf.subtract(1.0, y_pred)))
	return tf.reduce_sum(tf.subtract(part1, part2))

# Train the model
def train_model(labels, learning_rate):
	# Create the log loss function
	log_loss = create_log_loss(y, y_pred)

	# Define optimizer
	optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(log_loss)

	# Initialize the global variables defined earlier
	init = tf.compat.v1.global_variables_initializer()

	# Starting the TensorFlow session
	with tf.compat.v1.Session() as sess:

		# Initialize the variables
		sess.run(init)

		# Iterate through each epoch
		for epoch in range(num_epochs):

			# Run each sample through the optimizer
			for (_x1, _x2, _x3, _x4, _y) in zip(sepal_lengths_train, sepal_widths_train, petal_lengths_train, petal_widths_train, labels):
				sess.run(optimizer, feed_dict = {x1: _x1, x2: _x2, x3: _x3, x4: _x4, y: _y})

			# Calculate the loss during each epoch
			l = sess.run(log_loss, feed_dict = {x1: _x1, x2: _x2, x3: _x3, x4: _x4, y: _y})

			# Display the results every 10 epochs
			if (epoch + 1) % 10 == 0:
				print("Epoch: ", (epoch + 1), "loss = ", l, " w1 = ", sess.run(w1),
				 " w2 = ", sess.run(w2), " w3 = ", sess.run(w3), " w4 = ", sess.run(w4), " b = ", sess.run(b))

		# Store necessary values to be used outside of the session
		weight1 = sess.run(w1)
		weight2 = sess.run(w2)
		weight3 = sess.run(w3)
		weight4 = sess.run(w4)
		bias = sess.run(b)

		return {"w1": weight1, "w2": weight2, "w3": weight3, "w4": weight4, "b": bias}

# Train the models that determine whether or not a sample is each of the three species
setosa_model = train_model(is_setosa_train, learning_rate_setosa)
versicolor_model = train_model(is_versicolor_train, learning_rate_versicolor)
virginica_model = train_model(is_virginica_train, learning_rate_virginica)

# Evaluate the probability a sample is a certain species given a trained model
def evaluate_probability(model, sample):
	linear_val = ((model["w1"] * sample["sepal_length"]) + (model["w2"] * sample["sepal_width"]) + 
		(model["w3"] * sample["petal_length"]) + (model["w4"] * sample["petal_width"])
		 + model["b"])

	return 1.0 / (1.0 + math.exp(-linear_val))

# Iterate through each test sample
for index, sample in test_df.iterrows():
	# Calculate the probability the test sample is each of the three species
	setosa_prob = evaluate_probability(setosa_model, sample)
	versicolor_prob = evaluate_probability(versicolor_model, sample)
	virginica_prob = evaluate_probability(virginica_model, sample)

	print("setosa: ", setosa_prob, "versicolor: ", versicolor_prob, "virginica: ", virginica_prob)

	# Decide what species the sample using the highest probability
	if (setosa_prob > versicolor_prob) and (setosa_prob > virginica_prob):
		test_df.at[index, "predicted_class"] = "Iris-setosa"
	elif versicolor_prob > virginica_prob:
		test_df.at[index, "predicted_class"] = "Iris-versicolor"
	else:
		test_df.at[index, "predicted_class"] = "Iris-virginica"

test_df["prediction_correct"] = (test_df["class"] == test_df["predicted_class"])

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(test_df)

accuracy = 0.0
for val in test_df["prediction_correct"]:
	if val:
		accuracy += 1.0
accuracy /= len(test_df)
accuracy *= 100.0

print("Model predicts species with", accuracy, "% accuracy")
