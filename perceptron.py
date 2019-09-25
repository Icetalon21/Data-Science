import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron
import numpy as np

breast_cancer_data = skdata.load_breast_cancer()
x = breast_cancer_data.data   #feature vector
y = breast_cancer_data.target  #1's & 0's

model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)

#trains our perceptron model
model.fit(x, y)

#Predict the class.labels
predictions = model.predict(x)  #Examine this array, you will see 0's and 1's

print(predictions)

#if y[i] == prediction[i] then 1, else 0
#then su, and divide by N

#Check our accuracy
scores = np.where(predictions == y, 1, 0)
mean = np.mean(scores)
print(mean)

#Create a random training (90 percent) and testing (10 percent) split
idx = np.random.permutation(x.shape[0]) #creates permutation of all indicies
#print(idx)

split_idx = int(0.90*x.shape[0])
train_idx, test_idx = idx[:split_idx], idx[split_idx:]

#Select the examples from x and y to construct our training and testing sets
x_train, y_train = x[train_idx, :], y[train_idx]
x_test, y_test = x[test_idx, :], y[test_idx]

print(x_train.shape[0], x_test.shape[0])

model.fit(x_train, y_train)
scores_train = model.score(x_train, y_train)
print(scores_train)
scores_test = model.score(x_test, y_test)
print(scores_test)
