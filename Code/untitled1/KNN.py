#随机数
import numpy as np
#绘图函数
import matplotlib.pyplot as plt
#颜色库
from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
# 导入一些数据
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# 我们只取前两个特征。我们可以避免这种丑陋
# slicing by using a two-dim dataset
# 使用两个模糊数据集进行切片
X = iris.data[:, :2]
y = iris.target

# step size in the mesh
# 网格中的步长
h = .02

# Create color maps #创建彩色地图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    #我们创建一个邻居分类器的实例并适合数据。
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    # Plot the decision boundary. For that, we will assign a color to each
    # 绘制决策边界。为此，我们将为每个对象分配一个颜色
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    #点在网格中[x_min, x_max]x[y_min, y_max]。
    # 计算每个特征向量的最大值和最小值
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    #将结果放入颜色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    #也标出训练点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))
plt.show()
