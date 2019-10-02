import numpy as np
import sklearn.datasets as skdata
from sklearn.linear_model import LinearRegression

boston_housing_data = skdata.load_boston()
x = boston_housing_data.data
y = boston_housing_data.target

print(x.shape) #506, 13
print(y.shape) #506,

#Split the data 90 percent for training, 10 percent for testing
split_idx = int(0.90*x.shape[0])
x_train,y_train = x[:split_idx, :], y[:split_idx]
x_test, y_test = x[split_idx:, :], y[split_idx:]

#Sets up our Linear Regression Model
model = LinearRegression()

#Trains our regression model
model.fit(x_train, y_train)

#Make predictions
predictions = model.predict(x_test)

print("predictions are", predictions)

#meaduring regression error
def mean_squared_error(y_hat, y):
    #y_hat is your predictions
    return np.mean((y_hat-y)**2)

mse = mean_squared_error(predictions, y_test)

print("mse is", mse) #should be around 10.8~

#LINEAR REGRESSION VIA NORMAL EQUATION
class LinearRegressionNormalEqn(object):

    def __init__(self):
        self.weights = None

    def  fit(self, x, y):
        x_transpose_x_inv = np.linalg.inv(np.matmul(x.T, x)) # x.T == np.transpose(x)
        self.weights = np.matmul(np.matmul(x_transpose_x_inv, x.T), y)
        #xTx = np.matmul(np.transpose(x), x)
        #xTx_inv = np.inv(xTx)
        #self.weights = np.matmul(np.matmul(xTx_inv, np.transpose(x), y))

    def predict(self, x):
        return np.matmul(x, self.weights)

#Sets up our Linear Regression model
model_norm_eqn = LinearRegressionNormalEqn()

#Trains our regression model
model_norm_eqn.fit(x_train, y_train)

#Make predictions
predictions_norm_eqn = model_norm_eqn.predict(x_test)

#Evaluate our model
mse_norm_eqn = mean_squared_error(predictions_norm_eqn, y_test) #Should be around 13

print("mse norm eqn is ", mse_norm_eqn)

class LinearRegressionSVD(object):

    def __init__(self):
        self.weights = None

    def fit(self, x, y):
        U, S, V_t = np.linalg.svd(x)
        padding = np.zeros([x.shape[0] -S.shape[0], S.shape[0]])
        S_pseudo = np.concatenate([np.diag(1.0/S), padding], axis =0).T
        X_pseudo = np.matmul(np.matmul(V_t.T, S_pseudo), U.T)
        self.weights = np.matmul(X_pseudo, y)

    def predict(self, x):
        return np.matmul(x, self.weights)

#Split the data 90 percent for training, 10 percent for testing
split_idx = int(0.90*x.shape[0])
x_train, y_train = x[:split_idx, :], y[:split_idx]
x_test, y_test = x[split_idx:, :], y[split_idx:]
#if N < d, issues, too huge... use pseudo inverse//less dangerous

#Sets up our Linear Regression model
model_svd = LinearRegressionSVD()

#Trains our regression model
model_svd.fit(x_train, y_train)

#Make predictions
predictions_svd = model_svd.predict(x_test)

#Evaluate our model
mse_svd = mean_squared_error(predictions_svd, y_test)

print("mse svd", mse_svd)
