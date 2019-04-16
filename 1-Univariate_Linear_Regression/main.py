from mlAlgorithms import *;
import matplotlib.pyplot as plt;
import numpy as np;
import pandas as pd;

# load training data
dataMatrix = pd.read_csv("ex1data1.txt", header=None);

# convert a pandas dataframe (df) to a numpy ndarray
dataMatrix = dataMatrix.values;

# display plot dataMatrix
# [:, 0] = [select all rows, first column]
#plt.plot(dataMatrix[:, 0], dataMatrix[:, 1], marker='x', linestyle='None', color='red', label='Training samples')
#plt.show()

''' Gradient Descent '''

print("Preparing data for Gradient Descent...");

# prepare matrix for the gradient Descent
# (add a column of ones as the fisrt column of the matrix (x0))
newDataMatrix = np.hstack((np.ones((dataMatrix.shape[0], 1)), dataMatrix));

# features
X = newDataMatrix[:, :-1];

# results
y = newDataMatrix[:, -1:];

# Gradient descent settings
iterations = 1500;  # number of iterations for the gradient descent
alpha = 0.01;       # learning rate

# initial values for theta0 and theta1
theta = np.zeros(shape=(2, 1));   # 2 rows by 1 column

# initial cost function results for the initial theta values
initialCost = costFunction(X, y, theta);

print("\nInitial Cost (for theta0 = %.5f, theta1 = %.5f):" % (theta[0], theta[1]));
print(initialCost);

print("\nGradient Descent is running...");

(theta, J_history, theta_history) = gradientDescentWithHypothesisGraph(X, y, theta, alpha, iterations);

print("\nTheta found through gradient descent:");
print(theta);

## display plot with training set data and the hypothesis function
xAxis = X[:, 1];    # select all lines and the second column
yAxiz = np.matmul(X, theta);

# turn interactive mode off
plt.ioff();

plt.plot(dataMatrix[:, 0], dataMatrix[:, 1], marker='x', linestyle='None', color='red', label='Training samples');
plt.plot(xAxis, yAxiz, marker='None', linestyle='-', color='blue', label='Hypothesis');
plt.title("Training set and hypothesis function");
plt.xlabel("Feature x1 - Population in 10000s");
plt.ylabel("Profit in $10000s");
plt.legend();
plt.show();
#plt.show(block=False)

## display cost evolution
plt.clf();
# np.linspace(0,iterations, iterations): arg1: first elem, arg2: last element, arg3: how many elements in total
plt.plot(np.linspace(0,iterations, iterations), J_history, marker='None', linestyle='-', color='blue', label='J(theta)');
plt.title("Number of iterations");
plt.xlabel("Cost function J");
plt.ylabel("Cost function evolution");
plt.legend();
plt.show();
