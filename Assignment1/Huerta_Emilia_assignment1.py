import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import Perceptron


"""
Name: Huerta, Emilia

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


  def fit(self, x, y):
    """
    Fits the model to x and y by updating the weight vector
    based on mis-classified examples for t iterations until convergence

    x : N x d feature vector
    y : N x 1 ground-truth label
    """


  def predict(self, x):
    """
    Predicts the label for each feature vector x

    x : N x d feature vector

    returns : N x 1 label vector
    """

    return np.zeros(x.shape[0])

  def score(self, x, y):
    """
    Predicts labels based on feature vector x and computes the mean accuracy
    of the predictions

    x : N x d feature vector
    y : N x 1 ground-truth label

    returns : double
    """

    return 0.0


if __name__ == '__main__':

  breast_cancer_data = skdata.load_breast_cancer()
  x = breast_cancer_data.data
  y = breast_cancer_data.target

  # 90 percent train, 10 percent test split
  split_idx = int(0.90*x.shape[0])
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
