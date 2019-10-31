import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


"""
Name: Doe, John (Please write names in <Last Name, First Name> format)

Collaborators: Doe, Jane (Please write names in <Last Name, First Name> format)

Collaboration details: Discussed <function name> implementation details with Jane Doe.

Summary:
Report your scores here. For example,

Results on Wisconsin breast cancer dataset using scikit-learn Logistic Regression model
Training set mean accuracy: 0.9590
Testing set mean accuracy: 0.9649
Results on digits 7 and 9 dataset using scikit-learn Logistic Regression model
Training set mean accuracy: 1.0000
Testing set mean accuracy: 0.9722
Results on Boston housing price dataset using scikit-learn Linear Regression model
Training set mean accuracy: 23.2335
Testing set mean accuracy: 10.8062
Results on diabetes dataset using scikit-learn Linear Regression model
Training set mean accuracy: 2991.9850
Testing set mean accuracy: 1735.9381
Results on Wisconsin breast cancer dataset using our Logistic Regression model
Training set mean accuracy: 0.0000
Testing set mean accuracy: 0.0000
Results on digits 7 and 9 dataset using our Logistic Regression model
Training set mean accuracy: 0.0000
Testing set mean accuracy: 0.0000
Results on Boston housing price dataset using our Linear Regression model
Training set mean accuracy: 617.1287
Testing set mean accuracy: 369.2698
Results on diabetes dataset using our Linear Regression model
Training set mean accuracy: 29088.9673
Testing set mean accuracy: 28946.6889
"""


"""
Implementation of our Gradient Descent optimizer for mean squared loss and logistic loss
"""
class GradientDescentOptimizer(object):
  def __init__(self):
    pass

  def __compute_gradients(self, w, x, y, loss_func):
    """
    Returns the gradient of the logistic, mean squared or half mean squared loss

    w : 1 x d weight vector
    x : N x d feature vector
    y : N x 1 ground-truth label
    loss_func : loss type either 'logistic','mean_squared', or 'half_mean_squared'

    returns 1 x d gradients
    """
    if loss_func == 'logistic':
      return 0.0
    elif loss_func == 'mean_squared':
      return 0.0
    elif loss_func == 'half_mean_squared':
      return 0.0
    else:
      raise ValueError('Supported losses: logistic, mean_squared, or half_mean_squared')

  def update(self, w, x, y, alpha, loss_func):
    """
    Updates the weight vector based on logistic, mean squared or half mean squared loss

    w : 1 x d weight vector
    x : N x d feature vector
    y : N x 1 ground-truth label
    alpha : learning rate
    loss_func : loss type either 'logistic','mean_squared', or 'half_mean_squared'

    returns 1 x d weights
    """

    return w

"""
Implementation of our Logistic Regression model for binary classification
trained using Gradient Descent
"""
class LogisticRegressionGradientDescent(object):
  def __init__(self):
    # Define private variables
    self.__weights = None
    self.__optimizer = GradientDescentOptimizer()

  def fit(self, x, y, t, alpha, epsilon):
    """
    Fits the model to x and y by updating the weight vector
    using gradient descent

    x : N x d feature vector
    y : N x 1 ground-truth label
    t : number of iterations to train
    alpha : learning rate
    epsilon : threshold for stopping condition
    """
    self.__weights = np.random.rand(1, x.shape[1])

    # Gradient descent loop
    for i in range(int(t)):
      #sigmoid fuction
      y_pred_train = self.sigmoid(np.dot(x, self.__weights.T))
      dw = np.dot((y_pred_train - y).T, x) / x.shape[0]
      self.__weights = self.__weights - alpha * dw

  def sigmoid(self, x):
    return 1/ (1 + np.power(np.e, -x))

  def predict(self, x):
    """
    Predicts the label for each feature vector x

    x : N x d feature vector

    returns : N x 1 label vector
    """
    result = self.sigmoid(np.dot(x, self.__weights.T)).squeeze()
    result = np.array([0 if i < 0.5 else 1 for i in result])
    return result


  def score(self, x, y):
    """
    Predicts labels based on feature vector x and computes the mean accuracy
    of the predictions

    x : N x d feature vector
    y : N x 1 ground-truth label

    returns : double
    """
    h_x = self.predict(x)
    return np.mean(h_x == y)
"""
Implementation of our Linear Regression model trained using Gradient Descent
"""
class LinearRegressionGradientDescent(object):
  def __init__(self):
    # Define private variables
    self.__weights = None
    self.__optimizer = GradientDescentOptimizer()

  def fit(self, x, y, t, alpha, epsilon):
    """
    Fits the model to x and y by updating the weight vector
    using gradient descent

    x : N x d feature vector
    y : N x 1 ground-truth label
    t : number of iterations to train
    alpha : learning rate
    epsilon : threshold for stopping condition
    """
    self.__weights = np.random.rand(1, x.shape[1])

    # Gradient descent loop
    for i in range(int(t)):
      y_pred_train = np.dot(x, self.__weights.T)
      dw = np.dot((y_pred_train - y).T, x) / x.shape[0]
      self.__weights = self.__weights - alpha * dw

  def predict(self, x):
    """
    Predicts the label for each feature vector x

    x : N x d feature vector

    returns : N x 1 label vector
    """
    return np.dot(x, self.__weights.T).squeeze()

  def score(self, x, y):
    """
    Predicts labels based on feature vector x and computes the
    mean squared loss of the predictions

    x : N x d feature vector
    y : N x 1 ground-truth label

    returns : double
    """
    #make a prediction for all x
    h_x = self.predict(x) #[N,]
    mse = np.mean((h_x - y) **2)
    return mse

def mean_squared_error(y_hat, y):
  """
    Computes the mean squared error

    y_hat : N x 1 predictions
    y : N x 1 ground-truth label

    returns : double
    """
  return np.mean((y_hat-y)**2)

def format_data(x_train, y_train, x_test, y_test):
  x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
  y_train = np.expand_dims(y_train, axis=1)
  x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
  y_test = np.expand_dims(y_test, axis=1)

  return x_train, y_train, x_test, y_test

if __name__ == '__main__':

  # Loads breast cancer data with 90% training, 10% testing split
  breast_cancer_data = skdata.load_breast_cancer()
  x_cancer = breast_cancer_data.data
  y_cancer = breast_cancer_data.target

  split_idx = int(0.90*x_cancer.shape[0])
  x_cancer_train, y_cancer_train = x_cancer[:split_idx, :], y_cancer[:split_idx]
  x_cancer_test, y_cancer_test = x_cancer[split_idx:, :], y_cancer[split_idx:]

  # some preprocessing
  scaler = StandardScaler()
  x_cancer_train = scaler.fit_transform(x_cancer_train)
  x_cancer_test = scaler.transform(x_cancer_test)
  x_cancer_train, y_cancer_train, x_cancer_test, y_cancer_test = \
    format_data(x_cancer_train, y_cancer_train, x_cancer_test, y_cancer_test)

  # Loads 7 and 9 from digits data with 90% training, 10% testing split
  digits_data = skdata.load_digits()
  x_digits = digits_data.data
  y_digits = digits_data.target

  idx_79 = np.where(np.logical_or(y_digits == 7, y_digits == 9))[0]
  x_digits79 = x_digits[idx_79, :]
  y_digits79 = y_digits[idx_79]
  y_digits79[np.where(y_digits79 == 7)[0]] = 0
  y_digits79[np.where(y_digits79 == 9)[0]] = 1

  split_idx = int(0.90*x_digits79.shape[0])
  x_digits79_train, y_digits79_train = x_digits79[:split_idx, :], y_digits79[:split_idx]
  x_digits79_test, y_digits79_test = x_digits79[split_idx:, :], y_digits79[split_idx:]

  scaler = StandardScaler()
  x_digits79_train = scaler.fit_transform(x_digits79_train)
  x_digits79_test = scaler.transform(x_digits79_test)
  x_digits79_train, y_digits79_train, x_digits79_test, y_digits79_test = \
    format_data(x_digits79_train, y_digits79_train, x_digits79_test, y_digits79_test)

  # load data
  housing_data = skdata.load_boston()
  x_housing = housing_data.data
  y_housing = housing_data.target

  # split
  split_idx = int(0.90 * x_housing.shape[0])
  x_housing_train, y_housing_train = x_housing[:split_idx, :], y_housing[:split_idx]
  x_housing_test, y_housing_test = x_housing[split_idx:, :], y_housing[split_idx:]

  # some preprocessing
  scaler = StandardScaler()
  x_housing_train = scaler.fit_transform(x_housing_train)
  x_housing_test = scaler.transform(x_housing_test)
  x_housing_train, y_housing_train, x_housing_test, y_housing_test = \
    format_data(x_housing_train, y_housing_train, x_housing_test, y_housing_test)


  # Loads diabetes data with 90% training, 10% testing split
  diabetes_data = skdata.load_diabetes()
  x_diabetes = diabetes_data.data
  y_diabetes = diabetes_data.target

  split_idx = int(0.90*x_diabetes.shape[0])
  x_diabetes_train, y_diabetes_train = x_diabetes[:split_idx, :], y_diabetes[:split_idx]
  x_diabetes_test, y_diabetes_test = x_diabetes[split_idx:, :], y_diabetes[split_idx:]

  # some preprocessing
  scaler = StandardScaler()
  x_diabetes_train = scaler.fit_transform(x_diabetes_train)
  x_diabetes_test = scaler.transform(x_diabetes_test)
  x_diabetes_train, y_diabetes_train, x_diabetes_test, y_diabetes_test = \
    format_data(x_diabetes_train, y_diabetes_train, x_diabetes_test, y_diabetes_test)

  """
  Trains and tests Logistic Regression model from scikit-learn
  """
  # Trains scikit-learn Logistic Regression model on Wisconsin cancer data
  scikit_logistic_cancer = LogisticRegression(solver='liblinear')
  scikit_logistic_cancer.fit(x_cancer_train, y_cancer_train.squeeze())
  print('Results on Wisconsin breast cancer dataset using scikit-learn Logistic Regression model')
  # Test model on training set
  scikit_scores_cancer_train = scikit_logistic_cancer.score(x_cancer_train, y_cancer_train.squeeze())
  print('Training set mean accuracy: {:.4f}'.format(scikit_scores_cancer_train))
  # Test model on testing set
  scikit_scores_cancer_test = scikit_logistic_cancer.score(x_cancer_test, y_cancer_test.squeeze())
  print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_cancer_test))

  # Trains scikit-learn Logistic Regression model on digits 7 and 9 data
  scikit_logistic_digits79 = LogisticRegression(solver='liblinear')
  scikit_logistic_digits79.fit(x_digits79_train, y_digits79_train.squeeze())
  print('Results on digits 7 and 9 dataset using scikit-learn Logistic Regression model')
  # Test model on training set
  scikit_scores_digits79_train = scikit_logistic_digits79.score(x_digits79_train, y_digits79_train.squeeze())
  print('Training set mean accuracy: {:.4f}'.format(scikit_scores_digits79_train))
  # Test model on testing set
  scikit_scores_digits79_test = scikit_logistic_digits79.score(x_digits79_test, y_digits79_test.squeeze())
  print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_digits79_test))

  """
  Trains and tests Linear Regression model from scikit-learn
  """
  # Trains scikit-learn Linear Regression model on Boston housing price data
  scikit_linear_housing = LinearRegression()
  scikit_linear_housing.fit(x_housing_train, y_housing_train)
  print('Results on Boston housing price dataset using scikit-learn Linear Regression model')
  # Test model on training set
  scikit_predictions_housing_train = scikit_linear_housing.predict(x_housing_train)
  scikit_scores_housing_train = mean_squared_error(scikit_predictions_housing_train, y_housing_train)
  print('Training set mean accuracy: {:.4f}'.format(scikit_scores_housing_train))
  # Test model on testing set
  scikit_predictions_housing_test = scikit_linear_housing.predict(x_housing_test)
  scikit_scores_housing_test = mean_squared_error(scikit_predictions_housing_test, y_housing_test)
  print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_housing_test))

  # Trains scikit-learn Linear Regression model on diabetes data
  scikit_linear_diabetes = LinearRegression()
  scikit_linear_diabetes.fit(x_diabetes_train, y_diabetes_train)
  print('Results on diabetes dataset using scikit-learn Linear Regression model')
  # Test model on training set
  scikit_predictions_diabetes_train = scikit_linear_diabetes.predict(x_diabetes_train)
  scikit_scores_diabetes_train = mean_squared_error(scikit_predictions_diabetes_train, y_diabetes_train)
  print('Training set mean accuracy: {:.4f}'.format(scikit_scores_diabetes_train))
  # Test model on testing set
  scikit_predictions_diabetes_test = scikit_linear_diabetes.predict(x_diabetes_test)
  scikit_scores_diabetes_test = mean_squared_error(scikit_predictions_diabetes_test, y_diabetes_test)
  print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_diabetes_test))


  """
  Trains and tests our Logistic Regression model trained with Gradient Descent
  """
  # Trains our Logistic Regression model on Wisconsin cancer data
  t_cancer = 5000
  alpha_cancer = .01
  epsilon_cancer = 1e-4
  our_logistic_cancer = LogisticRegressionGradientDescent()
  our_logistic_cancer.fit(
      x_cancer_train, y_cancer_train, t_cancer, alpha_cancer, epsilon_cancer)
  print('Results on Wisconsin breast cancer dataset using our Logistic Regression model')
  #print("y cancer train", y_cancer_train)
  # Test model on training set
  our_scores_cancer_train = our_logistic_cancer.score(x_cancer_train, y_cancer_train.squeeze())
  print('Training set mean accuracy: {:.4f}'.format(our_scores_cancer_train))
  # Test model on testing set
  our_scores_cancer_test = our_logistic_cancer.score(x_cancer_test, y_cancer_test.squeeze())
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_cancer_test))

  # Trains our Logistic Regression model on digits 7 and 9 data
  t_digits79 = 5000
  alpha_digits79 = .01
  epsilon_digits79 = 1e-4
  our_logistic_digits79 = LogisticRegressionGradientDescent()
  our_logistic_digits79.fit(
      x_digits79_train, y_digits79_train, t_digits79, alpha_digits79, epsilon_digits79)
  print('Results on digits 7 and 9 dataset using our Logistic Regression model')
  # print("x digits 79 train", x_digits79_train)
  # print("y didgits 79 train", y_digits79_train)
  # Test model on training set
  our_scores_digits79_train = our_logistic_digits79.score(x_digits79_train, y_digits79_train.squeeze())
  print('Training set mean accuracy: {:.4f}'.format(our_scores_digits79_train))
  # Test model on testing set
  our_scores_digits79_test = our_logistic_digits79.score(x_digits79_test, y_digits79_test.squeeze())
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_digits79_test))

  """
  Trains and tests our Linear Regression model trained using Gradient Descent
  """
  # Trains our Linear Regression model on Boston housing price data
  t_housing = 5000
  alpha_housing = .01
  epsilon_housing = 1e-4
  our_linear_housing = LinearRegressionGradientDescent()
  our_linear_housing.fit(
      x_housing_train, y_housing_train, t_housing, alpha_housing, epsilon_housing)
  print('Results on Boston housing price dataset using our Linear Regression model')
  # Test model on training set
  our_predictions_housing_train = our_linear_housing.predict(x_housing_train)
  our_scores_housing_train = mean_squared_error(our_predictions_housing_train, y_housing_train.squeeze())
  print('Training set mean accuracy: {:.4f}'.format(our_scores_housing_train))
  # Test model on testing set
  our_predictions_housing_test = our_linear_housing.predict(x_housing_test)
  our_scores_housing_test = mean_squared_error(our_predictions_housing_test, y_housing_test.squeeze())
  #print("our predictions", our_predictions_housing_test)
  #print("y housing test", y_housing_test)
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_housing_test))

  # Trains our Linear Regression model on diabetes data
  t_diabetes = 5000
  alpha_diabetes = .01
  epsilon_diabetes = 1e-4
  our_linear_diabetes = LinearRegressionGradientDescent()
  our_linear_diabetes.fit(
      x_diabetes_train, y_diabetes_train, t_diabetes, alpha_diabetes, epsilon_diabetes)
  print('Results on diabetes dataset using our Linear Regression model')
  # Test model on training set
  our_predictions_diabetes_train = our_linear_diabetes.predict(x_diabetes_train)
  our_scores_diabetes_train = mean_squared_error(our_predictions_diabetes_train, y_diabetes_train.squeeze())
  print('Training set mean accuracy: {:.4f}'.format(our_scores_diabetes_train))
  # Test model on testing set
  our_predictions_diabetes_test = our_linear_diabetes.predict(x_diabetes_test)
  our_scores_diabetes_test = mean_squared_error(our_predictions_diabetes_test, y_diabetes_test.squeeze())
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_diabetes_test))
