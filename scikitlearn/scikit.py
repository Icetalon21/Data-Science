import sklearn
import numpy as np
import sklearn.datasets as skdata
from matplotlib import pyplot as plt

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
#plt.show()


'''
fig.suptitle('Boston Housing Data')
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel('Housing Price')
ax.set_xlabel('Crime Rate ({})'.format(feat_names[0]))
ax.scatter(crime_rate, y)
#plt.show()
'''
'''
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Per Capita Crime rate")
ax1.set_ylabel('Housing Price')
ax1.set_xlabel('Crime Rate ({})'.format(feat_names[0]))
ax.scatter(crime_rate, y)

nitric_oxide = x[:, 4]
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Nitric Oxide Concentration (PP 10M)")
ax2.set_ylabel('Housing Price')
ax2.set_xlabel('Nitric Oxide ({})'.format(feat_names[4]))
ax2.scatter(nitric_oxide, y)
plt.show()
'''

'''
#VISUALIZING DATA PAGE 1

fig = plt.figure()
fig.suptitle('Boston Housing Data')

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Per Capita Crime Rate")
ax1.set_ylabel('Housing Price')
ax1.set_xlabel('Crime Rate ({})'.format(feat_names[0]))
ax1.scatter(crime_rate,y)
'''
nitric_oxide = x[:, 4]
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Nitric Oxide Concentration (PP 10M")
ax2.set_ylabel('Housing Price')
ax2.set_xlabel('Nitric Oxide ({})'.format(feat_names[4]))
ax2.scatter(nitric_oxide, y)
#plt.show()

'''
#VISUALIZING DATA PAGE 2
fig = plt.figure()
fig.suptitle('Boston Housing Data')
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel('Housing Price')
ax.set_xlabel('Features')

'''

observations = (crime_rate, nitric_oxide)
targets = (y, y,)
colors = ("blue", "red") # Crime rate will be blue and nitric oxide will be red
labels = (feat_names[0], feat_names[4])
markers =('o', '^')
for obs, tgt, col, lab, m in zip(observations, targets, colors, labels, markers):
    ax.scatter(obs, tgt, c=col, label=lab, marker=m)
    ax.legend(loc='upper right')
#plt.show()


def min_max_norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))
crime_rate_minmax = min_max_norm(crime_rate)
nitric_oxide_minmax = min_max_norm(nitric_oxide)
print(np.min(crime_rate_minmax), np.max(crime_rate_minmax))
print(np.min(nitric_oxide_minmax), np.max(nitric_oxide_minmax))



#MIN-MAX NORMALIZATION
fig = plt.figure()
fig.suptitle('Boston Housing Data')
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel('Housing Price')
ax.set_xlabel('Min-Max Norm Features')

observations_minmax = (crime_rate_minmax, nitric_oxide_minmax)
for obs, tgt, col, lab, m in zip(observations, targets, colors, labels, markers):
    ax.scatter(obs, tgt, c=col, label=lab, marker=m)
    ax.legend(loc='upper right')
plt.show()
