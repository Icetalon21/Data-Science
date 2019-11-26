import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata

#We will use the iris dataset
iris_dataset = skdata.load_iris()
X = iris_dataset.data # (150, 4)
y = iris_dataset.target

#Compute the mean
mu = np.mean(X, axis=0)

#Center the data
B = X - mu

#Compute the covariance matrix
C = np.matmul(B.T, B)/(B.shape[0]) #(4, 150) x (150, 4) => (4, 4)

#Eigen decomposition
S, V = np.linalg.eig(C)

#Select the top 3 dimentions
order = np.argsort(S)[::-1]
W = V[:, order][:, 0:3] #Transformation for projecting X to subspace

#Project our data
Z = np.matmul(B, W) #(150, 3)

#Let's visualize our new feature space
data_split = (Z[np.where(y==0)[0], :], Z[np.where(y==1)[0], :], Z[np.where(y==2)[0], :])
colors = ('blue', 'red', 'green')
labels = ('Setosa', 'Versicolor', 'Virginica')
markers = ('o', '^', '+')

fig = plt.figure()
fig.suptitle('Projected Iris Data')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

for z, c, l, m in zip(data_split, colors, labels, markers):
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
    ax.legend(loc='upper right')

plt.show()

#Recover our data
X_hat = np.matmul(Z, W.T)+mu
mse = np.mean((X-X_hat)**2) #0.005919048088406607

#Seems like we recovered our data pretty well
#Let's instead choose only two dimentions
W_2 = V[:, 0:2] #Transformation for projecting X to subspace

#Project our data
Z_2 = np.matmul(B, W_2)

X_hat_2 = np.matmul(Z_2, W_2.T)+mu
mse = np.mean((X-X_hat_2)**2) # 0.02534107393239825

#As we reduce more dimentions, we lose more information
