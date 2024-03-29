# Iris Flower Species Classification using Logistic Regression

The Iris flower data set is a multivariate data set introduced by British statistician Ronald Fisher in 1936. The data set consists of 50 samples from each of three different species: Iris-Setosa, Iris-Versicolor, and Iris-Virginica. Four features were measured from each sample: the length and width of the petal and sepal lengths.

The goal of this project is to train a model that is capable of predicting an unknown sample's species with high accuracy using only the four features mentioned previously. However, before we create a model, we must understand the data we are working with. Running *vizualize_data.py* produces scatter plots for each pair of distinct features.

![Feature Scatter Plots](/images/iris_plot.png)

## Iris-Setosa
![Setosa Summary Data](/images/setosa_data.png)

## Iris-Versicolor
![Versicolor Summary Data](/images/versicolor_data.png)

## Iris-Virginica
![Virginica Summary Data](/images/virginica_data.png)

Looking at the data, we can see that although there are three visible clusters, the Setosa cluster is far more isolated than Versicolor or Virginica. We can easily seperate the Setosa samples by using either of the petal features. For example, the maximum petal length for a Setosa is 1.9 cm while the minimum lengths for Versicolor and Virginica are 3 cm and 4.5 cm respectively. A decision tree with a single node could classify a sample as *"Setosa"* or *"Not Setosa"* with 100% probability.

The challenge lies in differentiating between the Versicolor and Virginica species. For each feature, about the highest quartile of the Versicolor data overlaps with the lowest quartile of the Virginica data. This may cause a decrease of confidence when analyzing samples that fall into this overlap area.

## Building the Model

The model I have chosen to use for this data is a logistic regression model. Logistic regression is a good choice for this problem since we are using multiple continuous variables to try and predict a categorical outcome. This model should have no trouble classifying Setosa samples and should be able to classify the other two species with fairly high accuaracy.

Before building the model, we must split the data into training and test sets. *preprocess_data.py* reads in the entire data set from *iris.txt* and randomly splits the data into a training set (80%) and a test set (20%), then writes the results into *iris_train.txt* and *iris_test.txt*. This code should not be run more than once to avoid mixing the training data with the test data.
