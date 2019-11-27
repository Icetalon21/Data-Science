import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets as skdata
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# We will use the iris dataset
iris_dataset = skdata.load_iris()
X = iris_dataset.data # (150, 4)
y = iris_dataset.target

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_hat = kmeans.labels_

# Visualize by projecting to lower dimensions
Z = PCA(n_components=3).fit_transform(X)

data_split = (Z[np.where(y_hat == 0)[0], :],
              Z[np.where(y_hat == 1)[0], :], Z[np.where(y_hat == 2)[0], :])
colors = ('blue', 'red', 'green')
labels = ('Setosa', 'Versicolour', 'Virginica')
markers = ('o', '^', '+')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
for z, c, l, m in zip(data_split, colors, labels, markers):
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
    ax.legend(loc='upper right')

# Let’s compare it to groundtruth labels
data_split_kmeans = (Z[np.where(y_hat == 0)[0], :],
                     Z[np.where(y_hat == 1)[0], :], Z[np.where(y_hat == 2)[0], :])
data_split_gt = (Z[np.where(y == 0)[0], :],
                 Z[np.where(y == 1)[0], :], Z[np.where(y == 2)[0], :])

data_splits = [data_split_kmeans, data_split_gt]
plot_titles = ['Partition by k-Means', 'Groundtruth']
fig = plt.figure()
for i in range(len(data_splits)):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.set_title(plot_titles[i])
    for z, c, l, m in zip(data_splits[i], colors, labels, markers):
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
        ax.legend(loc='upper right')
plt.show()

# Let’s look at what happens if we set K to 5
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_hat = kmeans.labels_ # Can be one of {0, 1, 2, 3, 4}

data_split_kmeans = [
 Z[np.where(y_hat == i)[0], :] for i in range(5)]
data_split_gt = [Z[np.where(y == 0)[0], :],
 Z[np.where(y == 1)[0], :], Z[np.where(y == 2)[0], :]]

colors_gt = ('blue', 'red', 'green')
colors_kmeans = ('blue', 'red', 'green', 'yellow', 'magenta')

labels_gt = ('Setosa', 'Versicolour', 'Virginica')
labels_kmeans = ('0', '1', '2', '3', '4')

markers_gt = ('o', '^', '+')
markers_kmeans = ('o', '^', '+', '*', 'd')

data_splits = [data_split_kmeans, data_split_gt]
plot_titles = ['k-Means k=5', 'Groundtruth']
colors = [colors_kmeans, colors_gt]
labels = [labels_kmeans, labels_gt]
markers = [markers_kmeans, markers_gt]
fig = plt.figure()

for i in range(len(data_splits)):
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.set_title(plot_titles[i])
    for z, c, l, m in zip(data_splits[i], colors[i], labels[i], markers[i]):
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, label=l, marker=m)
        ax.legend(loc='upper right')

plt.show()

# Let’s see what happens if we try different initializations
X = np.random.rand(1500, 2)

data_splits = []
for i in range(4):
    kmeans = KMeans(n_clusters=2, init='random', n_init=1)
    kmeans.fit(X)
    y_hat = kmeans.labels_ # Can be one of {0, 1}
    data_splits.append([X[np.where(y_hat == i)[0], :] for i in range(2)])
colors = ('blue', 'red')
labels = ('0', '1')
markers = ('o', '^')

plot_titles = ['k-Means rand1', 'k-Means rand2', 'k-Means rand3', 'k-Means rand4']
fig = plt.figure()
for i in range(len(data_splits)):
    ax = fig.add_subplot(1, 4, i+1)
    ax.set_title(plot_titles[i])
    for x, c, l, m in zip(data_splits[i], colors, labels, markers):
        ax.scatter(x[:, 0], x[:, 1], c=c, label=l, marker=m)
        ax.legend(loc='upper right')

plt.show()

