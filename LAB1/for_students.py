import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

def calculate_error(theta, x, y):
    X = np.c_[np.ones(len(x)), x]
    predicted_output = np.matmul(X, theta.reshape((-1,1)))
    loss = predicted_output - y.reshape((-1, 1))
    error = np.average(np.square(loss))
    return error

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()
#x_train = np.c_[x_train1, x_train1**2] # join arrays


y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()
#x_test = np.c_[x_test1, x_test1**2] # join arrays

# TODO: calculate closed-form solution
theta_best = np.zeros(len(x_train.reshape(-1,1)[0]) + 1) # random initialization
X = np.c_[np.ones(len(x_train)), x_train] # join arrays
mult_inversed = np.linalg.inv(np.matmul(X.T, X))
mult2 = np.matmul(mult_inversed, X.T)
theta_best = np.matmul(mult2, y_train)
print("theta1 = ", theta_best)

# TODO: calculate error
print("error1 = ", calculate_error(theta_best, x_test, y_test))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
mean = np.mean(x_train) # average value
std = np.std(x_train) # standart deviation (odchylenie standardowe populacji)
mean_y = np.mean(y_train)
std_y = np.std(y_train)
# The standard deviation indicates how much the values in a dataset differ from the mean (average) value of the dataset.
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
y_train = (y_train - mean_y) / std_y
y_test = (y_test - mean_y) / std_y

# TODO: calculate theta using Batch Gradient Descent
theta_best = np.random.rand(len(x_train.reshape(-1,1)[0]) + 1, 1) # random initialization
alpha = 0.01 # learning rate
n_iterations = 1000
instances_counter = len(x_train)
X = np.c_[np.ones(instances_counter), x_train]
for _ in range(n_iterations):
    gradients = 2/instances_counter * X.T.dot(X.dot(theta_best) - y_train.reshape((-1, 1))) # calculate gradients using #1.7
    theta_best = theta_best - alpha * gradients
    if _ % 100 == 0:
        
        print("error = ", calculate_error(theta_best, x_test, y_test))

theta_best = theta_best.reshape(-1)
print("theta2 = ", theta_best)

# TODO: calculate error
print("error2 = ", calculate_error(theta_best, x_test, y_test))

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
x = x * std + mean
y = y * std_y + mean_y
plt.plot(x, y)
x_test = x_test * std + mean
y_test = y_test * std_y + mean_y
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()