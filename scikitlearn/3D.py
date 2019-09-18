import sklearn
import numpy as np
import sklearn.datasets as skdata
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

boston_housing_data = skdata.load_boston()

print(boston_housing_data)

x = boston_housing_data.data

feat_names = boston_housing_data.feature_names
print(feat_names)

#print(boston_housing_data.DESCR)

y = boston_housing_data.target

#print(y[0],y[0, :])

crime_rate = x[:, 0]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(crime_rate, y)

nitric_oxide = x[:, 4]
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Nitric Oxide Concentration (PP 10M")
ax2.set_ylabel('Housing Price')
ax2.set_xlabel('Nitric Oxide ({})'.format(feat_names[4]))
ax2.scatter(nitric_oxide, y)

def standard_norm(x):
    return(x-np.mean(x))/np.std(x)

crime_rate_std = standard_norm(crime_rate)
nitric_oxide_std = standard_norm(nitric_oxide)
print(np.min(crime_rate_std), np.max(crime_rate_std))
print(np.min(nitric_oxide_std), np.max(nitric_oxide_std))

#STANDARD NORMALIZATION
fig = plt.figure()
fig.suptitle('Boston Housing Data')
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel('Housing Price')
ax.set_xlabel('Standard Norm Features')

observations_std = (crime_rate_std, nitric_oxide_std)
targets = (y, y)
colors = ("blue", "red") # Crime rate will be blue and nitric oxide will be red
labels = (feat_names[0], feat_names[4])
markers =('o', '^')
for obs, tgt, col, lab, m in zip(observations_std, targets, colors, labels, markers):
    ax.scatter(obs, tgt, c=col, label=lab, marker=m)
    ax.legend(loc='upper right')

#VISUALIZING IN 3D

fig = plt.figure()
fig.suptitle('Boston Housing Data')
ax = fig.add_subplot(1, 1, 1, projection= '3d')
ax.set_xlabel('Crime rate')
ax.set_ylabel('Nitric Oxide')
ax.set_zlabel('Housing Price')
ax.scatter(crime_rate_std, nitric_oxide_std, y, c='blue', marker='o')

plt.show()
