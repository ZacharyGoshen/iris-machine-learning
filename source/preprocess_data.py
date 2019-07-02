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

# Write the training and test sets to csv files
train_df.to_csv("./../data/iris_train.txt", index = False)
test_df.to_csv("./../data/iris_test.txt", index = False)