import numpy as np
import sklearn.datasets as skdata
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron

breast_cancer_data = skdata.load_breast_cancer()
x_cancer = breast_cancer_data.data
y_cancer = breast_cancer_data.target

# Split the data 90 percent for training, 10 percent for testing
split_idx = int(0.90*x_cancer.shape[0])
x_cancer_train, y_cancer_train = x_cancer[:split_idx, :], y_cancer[:split_idx]
x_cancer_test, y_cancer_test = x_cancer[split_idx:, :], y_cancer[split_idx:]

x_cancer_train, y_cancer_train = x_cancer[:split_idx, :], y_cancer[:split_idx]
x_cancer_test, y_cancer_test = x_cancer[split_idx:, :], y_cancer[split_idx:]

# Try out logistic regression
model_logistic = LogisticRegression()
model_logistic.fit(x_cancer_train, y_cancer_train)

model_logistic_score = model_logistic.score(x_cancer_test, y_cancer_test) # 0.9649122807017544
print("model logistic score", model_logistic_score)

# Compare with Perceptron
model_perceptron = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
model_perceptron.fit(x_cancer_train, y_cancer_train)

model_perceptron_score = model_perceptron.score(x_cancer_test, y_cancer_test) # 0.
print("model perceptron score", model_perceptron_score)

digits_data = skdata.load_digits()
x_digits = digits_data.data
y_digits = digits_data.target

idx_01 = np.where(np.logical_or(y_digits == 0, y_digits == 1))[0]
x_digits01 = x_digits[idx_01, :]
y_digits01 = y_digits[idx_01]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(np.reshape(x_digits[0, ...], [8, 8]))
plt.show()

split_idx = int(0.90*x_digits01.shape[0])
x_digits01_train, y_digits01_train = x_digits01[:split_idx, :], y_digits01[:split_idx]
x_digits01_test, y_digits01_test = x_digits01[split_idx:, :], y_digits01[split_idx:]
plt.show()

# x_digits01_train, y_digits01_train = x_digits01[:split_idx, :], y_digits01[:split_idx]
# x_digits01_test, y_digits01_test = x_digits01[split_idx:, :], y_digits01[split_idx:]

# Try out logistic regression
model_logistic = LogisticRegression()
model_logistic.fit(x_digits01_train, y_digits01_train)
model_logistic.score(x_digits01_test, y_digits01_test) # 1.0

# Compare with Perceptron
model_perceptron = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
model_perceptron.fit(x_digits01_train, y_digits01_train)

model_perceptron.score(x_digits01_test, y_digits01_test) # 1.0

digits_data = skdata.load_digits()
x_digits = digits_data.data
y_digits = digits_data.target

idx_79 = np.where(np.logical_or(y_digits == 7, y_digits == 9))[0]
x_digits79 = x_digits[idx_79, :]
y_digits79 = y_digits[idx_79]

split_idx = int(0.90*x_digits79.shape[0])
x_digits79_train, y_digits79_train = x_digits79[:split_idx, :], y_digits79[:split_idx]
x_digits79_test, y_digits79_test = x_digits79[split_idx:, :], y_digits79[split_idx:]

# Try out logistic regression
model_logistic = LogisticRegression()
model_logistic.fit(x_digits79_train, y_digits79_train)

model_logistic_score2 = model_logistic.score(x_digits79_test, y_digits79_test) # 0.9714285714285714
print("model logistic score 2 ", model_logistic_score2)

# Compare with Perceptron
model_perceptron = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
model_perceptron.fit(x_digits79_train, y_digits79_train)

model_perceptron_score2 = model_perceptron.score(x_digits79_test, y_digits79_test) # 0.9714285714285714
print("model perceptron score 2 ", model_perceptron_score2)
