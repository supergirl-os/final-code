

from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
from data_loader import data_loader
mnist = datasets.load_digits()
X = mnist.data
y = mnist.target

print(X.shape,y.shape)
pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.Spectral)
plt.show()
