#!/usr/bin/python3 -u

import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron

import logging
import argparse

"""
Name: Kitamura, Masao
Collaborators:
    Gambetta, Grant
    Martin, Imani
    Huerta, Emilia
Collaboration details: Discussed overall implementation details with all collaborators.
Summary:
Results using scikit-learn Perceptron model
Training set mean accuracy: 0.7715
Testing set mean accuracy: 0.7544
Results using our Perceptron model
Training set mean accuracy: 0.6133
Testing set mean accuracy:  0.7544
"""

# Parse command line options
parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                    default=0, help='debug mode')

args = parser.parse_args()

# Initialize logging
# logging.basicConfig(format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
logging.basicConfig(format='%(levelname)s [%(filename)s:%(lineno)4d %(funcName)s] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

log = logging.getLogger(__name__)
if args.debug:
    log.setLevel(logging.DEBUG)


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

        log.info("UPDATE > PREDICT > START ...")
        predictions = self.predict(x)
        log.info("UPDATE > PREDICT > END ...")

        # Prepend x with 0.5 to make this work
        log.info("BEFORE INSERT x.shape:" + str(x.shape))        # (512, 32) !!!
        x = np.insert(x, 0, 0.5, axis=1)
        log.info(" AFTER INSERT x.shape:" + str(x.shape))        # (512, 32) !!!
        # print(x[n, :])  # array length 31

        log.info("UPDATE > LOOP > START ...")
        log.info("predictions.shape[0] == loop n max:" + str(predictions.shape[0]))
        for n in range(predictions.shape[0]):
            # log.info("UPDATE > LOOP > ITERATION: " + str(n))
            # p_n = predictions[n]
            # y_n = y[n]
            if (predictions[n] != y[n]):
                # log.info("weights.shape: " + str(self.__weights.shape))  # (1,31)
                # log.info("x[n, :].shape: " + str(x[n, :].shape))  # (1,31)
                # print(x[n, :])  # array length 31
                # print(y[n])     # 1

                self.__weights = self.__weights + x[n, :] * y[n]
        log.info("UPDATE > LOOP > END ...")

    def fit(self, x, y):
        """
        Fits the model to x and y by updating the weight vector
        based on mis-classified examples for t iterations until convergence
        x : N x d feature vector
        y : N x 1 ground-truth label
        """
        log.info("FIT ...")

        log.info("x.shape:" + str(x.shape))
        log.info("x.shape[0]:" + str(x.shape[0]))
        log.info("x.shape[1]:" + str(x.shape[1]))
        w = self.__weights

        # Set the weights here (initialize)
        w = np.zeros(x.shape[1] + 1)
        w[0] = -1
        print("w.shape:", w.shape)  # (31,)

        print("EXPAND DIMS")
        w = np.expand_dims(w, axis=-1)
        print("w.shape:", w.shape)  # (31, 1)

        print("w transpose")
        w = np.transpose(w)
        print("w.shape:", w.shape)  # (1, 31)
        print(w)

        self.__weights = w

        print("w.T.shape:", w.T.shape)  # (31, 1)

        # Convert all zero y's to -1, because predictions will only be 1 or -1
        y = np.where(y == 0, -1, 1)
        print("y.shape:", y.shape)  # (512, )

        log.info("FIT > BEGIN LOOP:")
        prev_loss = 1  # start at max loss
        prev_weights = np.array(self.__weights)

        # https://stats.stackexchange.com/questions/255375/what-exactly-is-tol-tolerance-used-as-stopping-criteria-in-sklearn-models
        tol = 0.001

        for t in range(1000):
            log.info("FIT > LOOP ITERATION:" + str(t))

            log.info("self.weights: " + str(self.__weights))

            # Predict
            predictions = self.predict(x)
            log.info("predictions.shape:" + str(predictions.shape))  # (512, 1)
            # log.info("predictions:" + str(predictions))

            # Compute error (E(h))
            log.info("FIT > LOOP ITERATION: " + str(t) + " > COMPUTE ERROR")
            log.debug("y:" + str(y))
            indicators = np.where(predictions != y, 1.0, 0.0)
            log.info("indicators.shape: " + str(indicators.shape))
            loss = np.mean(indicators)

            print("loss:", loss)
            print("prev_loss:", prev_loss)

            if loss == 0:
                log.info("LOSS == ZERO, breaking...")
                break
            elif loss > prev_loss + tol:
                log.info("LOSS > PREV_LOSS + TOL, breaking...")
                self.__weights = np.array(prev_weights)
                break

            prev_weights = np.array(self.__weights)
            prev_loss = loss

            # Update (will also call predict on x, and compare to corresponding y)
            log.info("FIT > LOOP ITERATION:" + str(t) + " > UPDATE")
            self.__update(x, y)

    def predict(self, x):
        """
        Predicts the label for each feature vector x
        x : N x d feature vector
        returns : N x 1 label vector
        """

        log.info("PREDICT ...")
        log.info("x.shape: " + str(x.shape))        # (512, 30)
        log.info("x.shape[0]: " + str(x.shape[0]))  # 512

        # Add some padding (moving to "predict" per email)
        log.info("FIT > ADD PADDING ...")
        log.info("x.shape[0]: " + str(x.shape[0]))  #
        threshold = np.full(x.shape[0], 0.5)
        log.info("threshold.shape: " + str(threshold.shape))  # (512,)
        log.info("x.shape: " + str(x.shape))  # (512,30)
        # log.info("x:" + str(x))

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.column_stack.html
        log.info("FIT > COLUMN STACK: append threshold to x")
        # x = np.concatenate((threshold, x.T), axis=1) # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
        x = np.column_stack((threshold, x))
        log.info("x.shape: " + str(x.shape))  # (512, 31)

        w = self.__weights
        log.info("w.shape: " + str(w.shape))  # (1, 31)

        log.info("PREDICT > MATMUL ...")
        log.info("w.T.shape: " + str(w.T.shape))  # (31, 1)

        predictions = np.matmul(x, w.T)
        log.info("predictions.shape: " + str(predictions.shape))  # (57, 1)

        signs = np.sign(predictions)
        # log.info("signs.shape: " + str(signs.shape))  # (57, 1)
        # print("signs:", signs)

        return signs

    def score(self, x, y):
        """
        Predicts labels based on feature vector x and computes the mean accuracy
        of the predictions
        x : N x d feature vector
        y : N x 1 ground-truth label
        returns : double
        """
        # Slide 31
        predictions = model.predict(x)
        log.debug("predictions.shape: " + str(predictions.shape))
        log.debug("predictions: " + str(predictions))

        # Convert all zero y's to -1, because predictions will only be 1 or -1
        y = np.where(y == 0, -1, 1)
        log.debug("y.shape:" + str(y.shape))
        log.debug("y:" + str(y))
        scores = np.where(predictions == y, 1, 0)

        print("scores:", scores)

        mean = np.mean(scores)
        return mean


if __name__ == '__main__':

    # MAIN PROGRAM STARTS HERE
    log.info("*** MAIN PROGRAM STARTS HERE ***")

    breast_cancer_data = skdata.load_breast_cancer()
    x = breast_cancer_data.data    # data to learn
    y = breast_cancer_data.target  # classification labels

    log.info("x.shape (data):" + str(x.shape))  # (569 + 30)
    log.info("y.shape (target):" + str(y.shape))  # (569 +)
    # print(y)
    # raise SystemExit

    # 90 percent train, 10 percent test split
    split_idx = int(0.90*x.shape[0])
    log.info("split_idx:" + str(split_idx))  # 512
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    log.info("x_train.shape:" + str(x_train.shape))  # (569 + 30)
    log.info("x_test.shape:" + str(x_test.shape))    # (57 + 30)
    log.info("y_train.shape:" + str(y_train.shape))  # (569 +)
    log.info("y_test.shape:" + str(y_test.shape))    # (57 +)

    '''
    Trains and tests Perceptron model from scikit-learn
    '''
    log.info("SCI KIT LEARN PERCEPTRON")
    model = Perceptron(penalty=None, alpha=0.0, tol=1e-3)

    # Trains scikit-learn Perceptron model
    model.fit(x_train, y_train)
    log.info('Results using scikit-learn Perceptron model')

    # Test model on training set
    scores_train = model.score(x_train, y_train)
    log.info('Training set mean accuracy: {:.4f}'.format(scores_train))

    # Test model on testing set
    scores_test = model.score(x_test, y_test)
    log.info('Testing set mean accuracy: {:.4f}'.format(scores_test))

    '''
    Trains and tests our Perceptron model for binary classification
    '''
    log.info("MANUAL PERCEPTRON")
    model = PerceptronBinary()

    # Trains scikit-learn Perceptron model
    log.info("*** MAIN > FIT > START ...")
    # FIT calls PREDICT/UPDATE in a loop
    model.fit(x_train, y_train)
    log.info("*** MAIN > FIT > END")
    print('*** Results using our Perceptron model')

    # Test model on training set
    log.info("*** MAIN > SCORE TRAIN START ...")
    # SCORE calls PREDICT (on X) which uses UPDATED weights updated in the loop in FIT
    scores_train = model.score(x_train, y_train)
    log.info("*** MAIN > SCORE TRAIN END")
    print('*** Training set mean accuracy: {:.4f}'.format(scores_train))

    # Test model on testing set
    log.info("*** MAIN > SCORE TEST START ...")
    scores_test = model.score(x_test, y_test)
    log.info("*** MAIN > SCORE TEST END")
    print('*** Testing set mean accuracy: {:.4f}'.format(scores_test))
