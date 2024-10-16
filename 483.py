import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


class1 = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5]])
class2 = np.array([[0.5, 1.5], [1.5, 0.5], [2.5, 1.5]])


X = np.vstack((class1, class2))
y = np.hstack((np.zeros(len(class1)), np.ones(len(class2))))


fig, ax = plt.subplots(1, 3, figsize=(18, 6))


x_min, x_max = 0, 3
y_min, y_max = 0, 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))


for i, k in enumerate([1, 3, 5]):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)


    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax[i].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)


    ax[i].scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', s=100, edgecolor='k', label='Class 1', marker='o')
    ax[i].scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='none', s=100, edgecolor='blue', label='Class 2', marker='o')


    ax[i].annotate("", xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax[i].annotate("", xy=(x_min, y_max), xytext=(x_min, y_min), arrowprops=dict(arrowstyle="->", lw=1.5))


    ax[i].set_xlabel('x', fontsize=14)
    if i == 0:
        ax[i].set_ylabel('y', fontsize=14)


    ax[i].set_title(f'k={k}', fontsize=16)


    ax[i].set_xlim(x_min, x_max)
    ax[i].set_ylim(y_min, y_max)


plt.tight_layout()
plt.show()
