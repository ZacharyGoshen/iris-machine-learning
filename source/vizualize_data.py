import math

import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read in the Iris data
iris_df = pd.read_csv("./../data/iris.txt", sep = ",")

# Create additional features that report whether a sample belongs to each species
iris_df['is_setosa'] = (iris_df['class'] == "Iris-setosa").astype(float)
iris_df['is_versicolor'] = (iris_df['class'] == "Iris-versicolor").astype(float)
iris_df['is_virginica'] = (iris_df['class'] == "Iris-virginica").astype(float)

# Create a feature for our future predictions
iris_df["predicted_class"] = "No prediction"

# Split data into training and test sets
train_df, test_df = train_test_split(iris_df, test_size = 0.2)

# Split the train data into groups seperated by label
train_setosa = train_df.loc[train_df["is_setosa"] == 1.0]
train_versicolor = train_df.loc[train_df["is_versicolor"] == 1.0]
train_virginica = train_df.loc[train_df["is_virginica"] == 1.0]

# Create a 4 x 4 figure of subplots for each feature pair
fig, ax = plt.subplots(4, 4, sharex = "col", sharey = "row")

# Format the legend
red_dot = mlines.Line2D([], [], color='red', marker='.', linestyle='None', markersize=10, label='Iris-Setosa')
green_dot = mlines.Line2D([], [], color='green', marker='.', linestyle='None', markersize=10, label='Iris-Versicolor')
blue_dot = mlines.Line2D([], [], color='blue', marker='.', linestyle='None', markersize=10, label='Iris-Virginica')
ax[0, 0].legend(handles = [red_dot, green_dot, blue_dot], loc = "upper left")

# Iterate through each sublplot
for index in range(1, 17):
	# Calculate row and column index of subplot
	row = math.floor((index - 1) / 4)
	col = ((index - 1) % 4)

	# Decide which features will be shown in sublplot
	features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
	x_var = features[row]
	y_var = features[col]

	# Show feature labels on left and bottom subplots
	if (row == 3):
		ax[row, col].set_xlabel(features[col].replace("_", " ") + " (cm)")
	if (col == 0):
		ax[row, col].set_ylabel(features[row].replace("_", " ") + " (cm)")

	# If subplot does not have the same feature for both its features
	if (x_var != y_var):
		# Group the data based on species
		g1 = (np.array(train_setosa[x_var]), np.array(train_setosa[y_var]))
		g2 = (np.array(train_versicolor[x_var]), np.array(train_versicolor[y_var]))
		g3 = (np.array(train_virginica[x_var]), np.array(train_virginica[y_var]))

		# Color and label the groups
		data = (g1, g2, g3)
		colors = ("red", "green", "blue")
		groups = ("setosa", "versicolor", "virginica")

		# Plot the data in the subplot
		for data, color, group in zip(data, colors, groups):
			x, y = data
			ax[row, col].plot(y, x, "o", c = color, label = group, markersize = 3)

# Show all subplots
plt.show()