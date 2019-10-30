import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


"""
Name: Huerta, Emilia (Please write names in <Last Name, First Name> format)

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
    # bias = 0.5 * np.ones([x.shape[0], 1])
    #
    # x = np.concatenate([bias, x], axis=-1)
    x = np.concatenate([0.5 * np.ones([x.shape[0], 1]), x], axis=-1)
    gradients = np.zeros([x.shape[0], x.shape[1]])

    # h = wx + b

    # h = np.dot(x, w.T)
    #
    # if loss_func == 'logistic':
    #   p = 1 / (1 + np.exp(-h))
    #
    #   loss = y*np.log(p) + (1-y)*np.log(1-p)
    #
    #   #grad = np.dot((y - p).T, x)
    #   grad = np.dot((y - p).T, x) / len(x)
    #
    #   return grad * np.mean(loss)
    if loss_func == 'logistic':
      gradients = np.zeros(x.shape)
      for n in range(x.shape[0]):
        x_n = x[n, ...]

        h_x = np.dot(np.squeeze(w), x_n)
        gradients[n, :] = (-y[n] * x_n) / (1.0 + np.exp(y[n] * h_x))
        return np.mean(gradients, axis=0)
      '''
      gradients = np.zeros(x.shape)
      for n in range(x.shape[0]):
        x_n = n[n, :]

        h_x = np.dot(np.squeeze(w), x_n)
        gradients[n,:] = (-y[n] * x_n) / (1.0 + np.exp(y[n] * h_x))
        return np.mean(gradients, axis=0)
      '''

    elif loss_func == 'mean_squared':
      # loss = np.mean((y - h)**2 )
      # grad = 2 * np.dot((y - h).T, x) / len(x)
      #
      # return grad * loss
      for n in range (x.shape[0]):
        x_n = x[n, :]
        h_x_n = np.dot(np.squeeze(w), x_n) #wTxN
        #print("testing", h_x_n-y[n])
        gradients[n] = (h_x_n - y[n]) * x_n
      #I will have N gradients
      return 2 * np.mean(gradients, axis=0)


    # elif loss_func == 'half_mean_squared':
    #   loss = np.mean((y - h) ** 2) / 2
    #   grad = np.dot((y - h).T, x) / len(x)
    #
    #   return grad * loss

    else:
      raise ValueError('Supported losses: logistic, mean_squared, or half_mean_squared')


    ''''
    dL/dw = dL/dp * dp/dh * dh/dw
    dL/db = dL/dp * dp/dh * dh/db
    
    dh/dw = x
    dh/db = 1
    
    dL/dw = dL/dp * p(1-p) * x
    dL/db = dL/dp * p(1-p) * 1
    
    logistic:
    p = 1 / (1 + exp-pred)
    L = y*log(p) + (1 - y)*log(1-p) => Cross entropy loss
    
    dL/dw = (y - p) / p*(1-p) * p(1-p) = (y - p) * x
    
    
    Mean squared:
    L = mean( (y - p)^2 )
    dL = 2 * (-1)
    
    dL/dw = mean(L* (-2) * X)
    
    
    '''


  def update(self, w, x, y, alpha, loss_func):  #this is correct for sure
    """
    Updates the weight vector based on logistic, mean squared or half mean squared loss

    w : 1 x d weight vector
    x : N x d feature vector
    y : N x 1 ground-truth label
    alpha : learning rate
    loss_func : loss type either 'logistic','mean_squared', or 'half_mean_squared'

    returns 1 x d weights
    """
    #alpha * self.__comppute_gradients()
    #w -= lr * grad * error

    #NEW WEIGHT = (OLD WIGHT - ALPHA) * GRADIENT

    w = w - alpha * self.__compute_gradients(w, x, y, loss_func)
    return w

"""
Implementation of our Logistic Regression model for binary classification
trained using Gradient Descent
"""
class LogisticRegressionGradientDescent(object):
  def __init__(self):
    """
    t : number of iterations to train
    alpha : learning rate
    epsilon : threshold for stopping condition
    """
    # Define private variables
    self.__weights = None
    self.__optimizer = GradientDescentOptimizer()

  # def threshold_function(self, h):
  #     _h = h.copy()
  #     _h[_h > 0] = 1
  #     _h[_h < 0] = 0
  #
  #     return _h

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
    # self.t = t
    # self.alpha= alpha
    # self.epsilon = epsilon
    # # Define private variables
    # #self.__bias = None
    # self.__error = np.inf


    self.__weights = np.zeros(x.shape[1])
    self.__bias = np.zeros(1)
    for i in range(int(t)): #changed from self.t
      error_sum = 0
      for j in range(x.shape[0]):
        error_sum += self.__update(x[j], y[j])

      error = error_sum / x.shape[0]

      # loss > previous_loss - tol
      if (np.abs(self.__error - error) < self.tol):
        print("Converged at epcoh:", i + 1)
        break

      else:
        self.__error = error


  def predict(self, x):
    """
    Predicts the label for each feature vector x

    x : N x d feature vector

    returns : N x 1 label vector
    """
    h = np.dot(x, self.__weights) + self.__bias  # [a1, a2, a3] . [b1, b2, b3] = [a1*b1 + a2*b2 + a3*b3]
    return self.threshold_function(h)
    #return np.zeros(x.shape[0])

  def score(self, x, y): #correct
    """
    Predicts labels based on feature vector x and computes the mean accuracy
    of the predictions

    x : N x d feature vector
    y : N x 1 ground-truth label

    returns : double
    """
    y_pred = self.predict(x)

    scores = np.where(y == y_pred, 1, 0)
    mean_accuracy = np.mean(scores)
    return mean_accuracy

    #return 0.0

"""
Implementation of our Linear Regression model trained using Gradient Descent
"""
class LinearRegressionGradientDescent(object):
  def __init__(self):
    # Define private variables
    self.__weights = None
    self.__optimizer = GradientDescentOptimizer()

  def threshold_function(self, h):
      _h = h.copy()
      _h[_h > 0] = 1
      _h[_h < 0] = 0

      return _h

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
    #gradient descent algorithm (basically)
    #initialize the weights
    #w=0, [N, d+1]

    # self.t = t
    # self.alpha = alpha
    # self.epsilon = epsilon
    # # Define private variables
    # #self.__bias = None
    # self.__error = np.inf


    # self.__weights = np.zeros(x.shape[1])  #initializes the weights
    #
    # self.__bias = np.zeros(1)
    # for i in range(int(t)): #changed from self.t
    #   error_sum = 0
    #   for j in range(x.shape[0]):
    #     error_sum += self.__update(x[j], y[j])
    #
    #   error = error_sum / x.shape[0]
    #
    #   # loss > previous_loss - tol
    #   if (np.abs(self.__error - error) < self.epsilon):
    #     print("Converged at epcoh:", i + 1)
    #     break
    #
    #   else:
    #     self.__error = error


    self.__weights = np.zeros(x.shape[1] +1)
    self.__weights[0] = -1.0
    for i in range(int(t)):
      #predict N values
      h_x = self.predict(x)
      #compute the loss
      loss = np.mean((h_x-y)**2)
      w_i = self.__optimizer.update(self.__weights, x, y, alpha, loss_func='mean_squared')
      if loss == 0:
        break
      #loss is 0
        #break
      #Change in magnitude is small
      # ||w**(t+1) - w**(t)|| < eplison
      #the change in weights
      d_w = self.__weights - w_i
      mag_d_w = np.sqrt(np.sum(d_w**2))
      #save the weights
      self.__weights = w_i
      if mag_d_w < epsilon:
        break

  def predict(self, x):
    """
    Predicts the label for each feature vector x
      w = self.weights
    x : N x d feature vector

    returns : N x 1 label vector
    """
    #Assume x has size [N, d]
    # weight w " " [1, d + 1]
    #h(x) = w1x1 + w2x2 + .... wdxd - threshold
    x = np.concatenate([0.50*np.ones([x.shape[0], 1]), x], axis=-1)
    h_x = np.zeros(x.shape[0])
    for n in range(x.shape[0]):
      x_n = x[n,:] #single example of x
      x_n = np.expand_dims(x_n, axis=-1)
      #h_x[n] = np.dot(np.squeeze(self.__weights), np.squeeze(x))
      h_x[n] = np.dot(np.squeeze(self.__weights), x_n)
    #N predictions stored in h_x = wTx
    return h_x

    # h = np.dot(x, self.__weights) + self.__bias  # [a1, a2, a3] . [b1, b2, b3] = [a1*b1 + a2*b2 + a3*b3]
    # return self.threshold_function(h)

    #return np.zeros(x.shape[0])

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
    # y_pred = self.predict(x)
    #
    # scores = np.where(y == y_pred, 1, 0)
    # mean_accuracy = np.mean(scores)
    # return mean_accuracy

def mean_squared_error(y_hat, y):
  """
    Computes the mean squared error

    y_hat : N x 1 predictions
    y : N x 1 ground-truth label

    returns : double
    """
  return np.mean((y_hat-y)**2) #correct
  #https://stackoverflow.com/questions/39064684/mean-squared-error-in-python/39065217

if __name__ == '__main__':

  # Loads breast cancer data with 90% training, 10% testing split
  breast_cancer_data = skdata.load_breast_cancer()
  x_cancer = breast_cancer_data.data
  y_cancer = breast_cancer_data.target

  split_idx = int(0.90*x_cancer.shape[0])
  x_cancer_train, y_cancer_train = x_cancer[:split_idx, :], y_cancer[:split_idx]
  x_cancer_test, y_cancer_test = x_cancer[split_idx:, :], y_cancer[split_idx:]

  # Loads 7 and 9 from digits data with 90% training, 10% testing split
  digits_data = skdata.load_digits()
  x_digits = digits_data.data
  y_digits = digits_data.target

  idx_79 = np.where(np.logical_or(y_digits == 7, y_digits == 9))[0]
  x_digits79 = x_digits[idx_79, :]
  y_digits79 = y_digits[idx_79]

  split_idx = int(0.90*x_digits79.shape[0])
  x_digits79_train, y_digits79_train = x_digits79[:split_idx, :], y_digits79[:split_idx]
  x_digits79_test, y_digits79_test = x_digits79[split_idx:, :], y_digits79[split_idx:]

  # Loads Boston housing price data with 90% training, 10% testing split
  housing_data = skdata.load_boston()
  x_housing = housing_data.data
  y_housing = housing_data.target

  split_idx = int(0.90*x_housing.shape[0])
  x_housing_train, y_housing_train = x_housing[:split_idx, :], y_housing[:split_idx]
  x_housing_test, y_housing_test = x_housing[split_idx:, :], y_housing[split_idx:]


  # Loads diabetes data with 90% training, 10% testing split
  diabetes_data = skdata.load_diabetes()
  x_diabetes = diabetes_data.data
  y_diabetes = diabetes_data.target

  split_idx = int(0.90*x_diabetes.shape[0])
  x_diabetes_train, y_diabetes_train = x_diabetes[:split_idx, :], y_diabetes[:split_idx]
  x_diabetes_test, y_diabetes_test = x_diabetes[split_idx:, :], y_diabetes[split_idx:]

  """
  Trains and tests Logistic Regression model from scikit-learn
  """
  # Trains scikit-learn Logistic Regression model on Wisconsin cancer data
  scikit_logistic_cancer = LogisticRegression(solver='liblinear')
  scikit_logistic_cancer.fit(x_cancer_train, y_cancer_train)
  print('Results on Wisconsin breast cancer dataset using scikit-learn Logistic Regression model')
  # Test model on training set
  scikit_scores_cancer_train = scikit_logistic_cancer.score(x_cancer_train, y_cancer_train)
  print('Training set mean accuracy: {:.4f}'.format(scikit_scores_cancer_train))
  # Test model on testing set
  scikit_scores_cancer_test = scikit_logistic_cancer.score(x_cancer_test, y_cancer_test)
  print('Testing set mean accuracy: {:.4f}'.format(scikit_scores_cancer_test))

  # Trains scikit-learn Logistic Regression model on digits 7 and 9 data
  scikit_logistic_digits79 = LogisticRegression(solver='liblinear')
  scikit_logistic_digits79.fit(x_digits79_train, y_digits79_train)
  print('Results on digits 7 and 9 dataset using scikit-learn Logistic Regression model')
  # Test model on training set
  scikit_scores_digits79_train = scikit_logistic_digits79.score(x_digits79_train, y_digits79_train)
  print('Training set mean accuracy: {:.4f}'.format(scikit_scores_digits79_train))
  # Test model on testing set
  scikit_scores_digits79_test = scikit_logistic_digits79.score(x_digits79_test, y_digits79_test)
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
  t_cancer = 0.0
  alpha_cancer = 0.0
  epsilon_cancer = 0.0
  our_logistic_cancer = LogisticRegressionGradientDescent()
  our_logistic_cancer.fit(
      x_cancer_train, y_cancer_train, t_cancer, alpha_cancer, epsilon_cancer)
  print('Results on Wisconsin breast cancer dataset using our Logistic Regression model')
  # Test model on training set
  our_scores_cancer_train = our_logistic_cancer.score(x_cancer_train, y_cancer_train)
  print('Training set mean accuracy: {:.4f}'.format(our_scores_cancer_train))
  # Test model on testing set
  our_scores_cancer_test = our_logistic_cancer.score(x_cancer_test, y_cancer_test)
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_cancer_test))

  # Trains our Logistic Regression model on digits 7 and 9 data
  t_digits79 = 0.0
  alpha_digits79 = 0.0
  epsilon_digits79 = 0.0
  our_logistic_digits79 = LogisticRegressionGradientDescent()
  our_logistic_digits79.fit(
      x_digits79_train, y_digits79_train, t_digits79, alpha_digits79, epsilon_digits79)
  print('Results on digits 7 and 9 dataset using our Logistic Regression model')
  # Test model on training set
  our_scores_digits79_train = our_logistic_digits79.score(x_digits79_train, y_digits79_train)
  print('Training set mean accuracy: {:.4f}'.format(our_scores_digits79_train))
  # Test model on testing set
  our_scores_digits79_test = our_logistic_digits79.score(x_digits79_test, y_digits79_test)
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_digits79_test))

  """
  Trains and tests our Linear Regression model trained using Gradient Descent
  """
  # Trains our Linear Regression model on Boston housing price data
  '''
  t : number of iterations to train
  alpha : learning rate
  epsilon : threshold for stopping condition
  '''
  t_housing = 100
  alpha_housing = 1e-6
  epsilon_housing = 1e-8
  our_linear_housing = LinearRegressionGradientDescent()
  our_linear_housing.fit(
      x_housing_train, y_housing_train, t_housing, alpha_housing, epsilon_housing)
  print('Results on Boston housing price dataset using our Linear Regression model')
  # Test model on training set
  our_predictions_housing_train = our_linear_housing.predict(x_housing_train)
  our_scores_housing_train = mean_squared_error(our_predictions_housing_train, y_housing_train)
  print('Training set mean accuracy: {:.4f}'.format(our_scores_housing_train))
  # Test model on testing set
  our_predictions_housing_test = our_linear_housing.predict(x_housing_test)
  our_scores_housing_test = mean_squared_error(our_predictions_housing_test, y_housing_test)
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_housing_test))

  # Trains our Linear Regression model on diabetes data
  t_diabetes = 0.0
  alpha_diabetes = 0.0
  epsilon_diabetes = 0.0
  our_linear_diabetes = LinearRegressionGradientDescent()
  our_linear_diabetes.fit(
      x_diabetes_train, y_diabetes_train, t_diabetes, alpha_diabetes, epsilon_diabetes)
  print('Results on diabetes dataset using our Linear Regression model')
  # Test model on training set
  our_predictions_diabetes_train = our_linear_diabetes.predict(x_diabetes_train)
  our_scores_diabetes_train = mean_squared_error(our_predictions_diabetes_train, y_diabetes_train)
  print('Training set mean accuracy: {:.4f}'.format(our_scores_diabetes_train))
  # Test model on testing set
  our_predictions_diabetes_test = our_linear_diabetes.predict(x_diabetes_test)
  our_scores_diabetes_test = mean_squared_error(our_predictions_diabetes_test, y_diabetes_test)
  print('Testing set mean accuracy: {:.4f}'.format(our_scores_diabetes_test))
