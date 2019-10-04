#!/usr/bin/env python3

import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron

import logging
log = logging.getLogger(__name__)


"""
Name: Doe, John (Please write names in <Last Name, First Name> format)
Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)
Collaboration details: Discussed <function name> implementation details with Jane Doe.
Summary:
Report your scores here. For example,
Results using scikit-learn Perceptron model
Training set mean accuracy: 0.9199
Testing set mean accuracy: 0.9298
Results using our Perceptron model
Training set mean accuracy: 0.0000
Testing set mean accuracy: 0.0000
"""

"""
Implementation of our Perceptron model for binary classification
"""


class PerceptronBinary(object):
    def __init__(self):
        # Define private variables
        self.__weights = None

    def __update(self, x, y):
        """
        Update the weight vector during each training iteration
        x : N x d feature vector
        y : N x 1 ground-truth label
        """

        # self.__weights = self.__weights + y[n] * x

        predictions = self.predict(x)
        for n in range(predictions.shape[0]):
            if (predictions[n] != y[n]):
                self.__weights = self.__weights + x[n, :] * y[n]

    def fit(self, x, y):
        """
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence
        x : N x d feature vector
        y : N x 1 ground-truth label
        """
        print("FIT ...")

        print("x.shape:", x.shape)
        print("x.shape[0]:", x.shape[0])
        print("x.shape[1]:", x.shape[1])
        w = self.__weights

        # Set the weights here (initialize)
        w = np.zeros(x.shape[1] + 1)
        w[0] = -1
        print("w.shape:", w.shape)  # (31,)

        print("EXPAND DIMS")
        w = np.expand_dims(w, axis=-1)
        print("w.shape:", w.shape)  # (31, 1)

        print("W transpose")
        w = np.transpose(w)
        print("w.shape:", w.shape)  # (1, 31)
        print(w)

        self.__weights = w

        print("w.T.shape:", w.T.shape)  # (31, 1)

        # Add some padding
        print("FIT > ADD PADDING ...")
        print("x.shape[0]:", x.shape[0])  #
        threshold = np.full(x.shape[0], 0.5)
        print("threshold.shape:", threshold.shape)  # (512,)
        print("x.shape:", x.shape)  # (512,30)
        print("x:", x)

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html
        print("FIT > COLUMN STACK: append threshold to x")
        # x = np.concatenate((threshold, x.T), axis=1) # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
        x = np.column_stack((threshold, x))
        print("x.shape:", x.shape)  # (512, 31)

        y = np.where(y == 0, -1, 1)
        print("y.shape:", y.shape)  # (512, )

        print("FIT > BEGIN LOOP:")
        for t in range(2):
            print("FIT > LOOP ITERATION:", t)
            # Predict
            predictions = self.predict(x)

            # Compute error (E(h))
            indictators = np.where(predictions != y, 1.0, 0.0)
            loss = np.mean(indictators)

            # Update
            self.__update(x, y)

    def predict(self, x):
        """
        Predicts the label for each feature vector x
        x : N x d feature vector
        returns : N x 1 label vector
        """

        print("PREDICT ...")
        print("x.shape:", x.shape)        # (512, 31)
        print("x.shape[0]:", x.shape[0])  # 512

        row = x[0, :]
        print("row.shape:", row.shape)  # (31,)

        w = self.__weights
        print("w.shape:", w.shape)  # (1, 31)

        trans = np.transpose(w)
        print("trans.shape:", trans.shape)  # (513,)
        print("w.shape:", w.shape)  # (1, 31)

        print("MATMUL ...")
        print("x.shape:", x.shape)      # (513, 31)
        print("w.T.shape:", w.T.shape)  # (31, 1)
        predictions = np.matmul(x, w.T)
        print("predictions.shape:", predictions.shape)  #

        return predictions

    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions
        x : N x d feature vector
        y : N x 1 ground-truth label
        returns : double
        """

        # Add some padding
        print("SCORE > ADD PADDING ...")
        print("x.shape[0]:", x.shape[0])  #
        threshold = np.full(x.shape[0], 0.5)
        print("threshold.shape:", threshold.shape)  # (512,)
        print("x.shape:", x.shape)  # (512,30)
        print("x:", x)

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html
        print("SCORE > COLUMN STACK: append threshold to x")
        # x = np.concatenate((threshold, x.T), axis=1) # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
        x = np.column_stack((threshold, x))
        print("x.shape:", x.shape)  # (512, 31)

        # x at this point is not the right shape, therefore we copy/pasted the above code
        # ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, ...
        # ... with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 31 is different from 30)

        # Slide 31
        predictions = model.predict(x)
        print("predictions.shape:", predictions.shape)
        # print("predictions:", predictions)
        scores = np.where(predictions == y, 1, 0)
        mean = np.mean(scores)
        return mean


if __name__ == '__main__':

    # RANDOM TESTS TO LEARN NP STUFF

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])

    print("a.shape:", a.shape)
    print("b.shape:", b.shape)
    print("b.T.shape:", b.T.shape)

    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
    c = np.concatenate((a, b), axis=0)
    print(c)

    c = np.concatenate((a, b.T), axis=1)
    print(c)

    d = np.array([[5, 6], [7, 8]])
    print("d:", d)
    print("d.shape:", d.shape)
    dot = np.dot(a, d)
    print("dot:", dot)

    # d = np.array([[5, 6], [7, 8], [9, 10]])
    # print("d:", d)
    # print("d.shape:", d.shape)
    # dot = np.dot(a, d)
    # print("dot:", dot)

    d = np.array([[5, 6, 9], [7, 8, 10]])
    print("d:", d)
    print("d.shape:", d.shape)
    dot = np.dot(a, d)
    print("dot:", dot)

    # MAIN PROGRAM STARTS HERE
    log.info("*** MAIN PROGRAM STARTS HERE ***")

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data    # data to learn
    y = breast_cancer_data.target  # classification labels

    print("x.shape (data):", x.shape)  # (569, 30)
    print("y.shape (target):", y.shape)  # (569,)
    # print(y)
    # raise SystemExit

    # 90 percent train, 10 percent test split
    split_idx = int(0.90*x.shape[0])
    print("split_idx:", split_idx)  # 512
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("x_train.shape:", x_train.shape)  # (569, 30)
    print("x_test.shape:", x_test.shape)    # (57, 30)
    print("y_train.shape:", y_train.shape)  # (569,)
    print("y_test.shape:", y_test.shape)    # (57,)

    '''
    Trains and tests Perceptron model from scikit-learn
    '''
    model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)

    # Trains scikit-learn Perceptron model
    # FIT calls PREDICT/UPDATE in a loop
    model.fit(x_train, y_train)
    print('Results using scikit-learn Perceptron model')

    # Test model on training set
    # SCORE calls PREDICT on UPDATED weights updated in a loop in FIT
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
    print("MAIN > FIT > START ...")
    model.fit(x_train, y_train)
    print("MAIN > FIT > DONE")
    print('Results using our Perceptron model')

    # Test model on training set
    print("MAIN > FIT > SCORE TRAIN START ...")
    scores_train = model.score(x_train, y_train)
    print("MAIN > FIT > SCORE TRAIN END")
    print('Training set mean accuracy: {:.4f}'.format(scores_train))

    # Test model on testing set
    print("MAIN > FIT > SCORE TEST START ...")
    scores_test = model.score(x_test, y_test)
    print("MAIN > FIT > SCORE TEST END")
    print('Testing set mean accuracy: {:.4f}'.format(scores_test))
