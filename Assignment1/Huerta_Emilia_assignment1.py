import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron

"""
Name: Huerta, Emilia
​
Collaborators:
    Gambetta, Grant
    Kitamura, Masao
    Martin, Imani
​
Collaboration details: Discussed __update(), fit(), predict(), & score() together
​
Summary:
Report your scores here. For example,
​
Results using scikit-learn Perceptron model
Training set mean accuracy: 0.9199
Testing set mean accuracy: 0.9298
Results using our Perceptron model
Training set mean accuracy: 0.9219
Testing set mean accuracy: 0.9474
"""

"""
Implementation of our Perceptron model for binary classification
"""


class PerceptronBinary(object):
    def __init__(self, max_ite=100, tol=1e-4, eta=0.1):
        self.max_iter = max_ite;
        self.learning_rate = eta
        self.tol = tol
        # Define private variables
        self.__weights = None
        self.__bias = None
        self.__error = np.inf

    def threshold_function(self, h):
        _h = h.copy()
        _h[_h > 0] = 1
        _h[_h < 0] = 0

        return _h


    def __update(self, x, y):
        """
        Update the weight vector during each training iteration
​
        x : N x d feature vector
        y : N x 1 ground-truth label
        """
        # w = w + x*y
        # shifts the weight matrix
        '''
        predictions = self.predict(x)
        for i in range(predictions.shape[0]):
            if predictions[i] != y[i]:
                self.__weights += x[i, :] * y[i]
        '''
        # Forward pass
        y_pred = self.predict(x)
        error = y - y_pred

        self.__weights += self.learning_rate * error * x
        self.__bias += self.learning_rate * np.sum(error)

        return np.abs(error)


    def fit(self, x, y):
        """
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence
    ​
        x : N x d feature vector
        y : N x 1 ground-truth label

        """
        '''
        self.__weights = np.zeros(x.shape[0] + 1)
        self.__weights[0] = -1
        threshold = np.full(0.5, x.shape[0])
        x = np.concatenate(threshold, x, axis = 1)
        y = np.where(y == 0, -1, 1)
        for i in range(1000): 
            predictions = self.predict(x)
            loss = np.mean(np.where(predictions != y, 1.0, 0.0))
            self.__update(x, y)
        '''
        self.__weights = np.zeros(x.shape[1])
        self.__bias = np.zeros(1)
        for i in range(self.max_iter):
            error_sum = 0
            for j in range(x.shape[0]):
                error_sum += self.__update(x[j], y[j])

            error = error_sum / x.shape[0]

            # loss > previous_loss - tol
            if(np.abs(self.__error - error) < self.tol ):
                print("Converged at epcoh:", i+1)
                break

            else:
                self.__error = error



    def predict(self, x):
        """
        Predicts the label for each feature vector x
    ​
        x : N x d feature vector
    ​
        returns : N x 1 label vector
        """
        '''
        scores = np.dot(np.transpose(self.__weights), x)
        return np.sign(scores)
        '''
        """
        weight = self.__weights
        weight = np.expand_dims(weight, axis=-1)
        transpose = np.transpose(weight)
        # print('w shape: ', weight.shape)
        # print('x shape: ', x.shape)
        # print('tranpose shape: ', transpose.shape)
        return np.sign(np.matmul(x, weight))
        """
        # x(datapoints, features)
        h = np.dot(x, self.__weights) + self.__bias # [a1, a2, a3] . [b1, b2, b3] = [a1*b1 + a2*b2 + a3*b3]
        return self.threshold_function(h)


    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions
    ​
        x : N x d feature vector
        y : N x 1 ground-truth label
    ​
        returns : double
        """
        '''
        threshold = np.full(x.shape[0], 0.5)
        x = np.column_stack((threshold, x))
        predictions = model.predict(x)
        scores = np.where(y == predictions, 1, 0)
        mean_accuracy = np.mean(scores)
        return mean_accuracy
        '''
        y_pred = self.predict(x)

        scores = np.where(y == y_pred, 1, 0)
        mean_accuracy = np.mean(scores)
        return mean_accuracy



if __name__ == '__main__':

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data
    y = breast_cancer_data.target

    # 90 percent train, 10 percent test split
    split_idx = int(0.90 * x.shape[0])
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    '''
    Trains and tests Perceptron model from scikit-learn
'''
    model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)
    # Trains scikit-learn Perceptron model
    model.fit(x_train, y_train)
    print('Results using scikit-learn Perceptron model')
    # Test model on training set
    scores_train = model.score(x_train, y_train)
    print('Training set mean accuracy: {:.4f}'.format(scores_train))
    # Test model on testing set
    scores_test = model.score(x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))

    '''
    Trains and tests our Perceptron model for binary classification
'''
    model = PerceptronBinary()
    # Trains scikit-learn Perceptron model
    model.fit(x_train, y_train)
    print('Results using our Perceptron model')
    # Test model on training set
    scores_train = model.score(x_train, y_train)
    print('Training set mean accuracy: {:.4f}'.format(scores_train))
    # Test model on testing set
    scores_test = model.score(x_test, y_test)
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))
