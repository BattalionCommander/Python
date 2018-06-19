#随机数
#绘图函数
#颜色库
#数据库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets




# import some data to play with
# 导入一些数据
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# 我们只取前两个特征。我们可以避免这种丑陋
# slicing by using a two-dim dataset
# 使用两个模糊数据集进行切片
#iris.data自变量矩阵
#iris.target
X = iris.data[:, :2]

y = iris.target

# step size in the mesh
# 网格中的步长
h = .02

#分类距离
n_neighbors = 15

#创建彩色地图,cmap_light用作背景的颜色，cmap_bold用作点的颜色
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#weights 平均权重，uniform一般平均，  distance与距离成反比
for weights in ['uniform', 'distance']:

    # we create an instance of Neighbours Classifier and fit the data.
    #我们创建一个临近分类模型并结合数据。
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # 绘制决策边界。为此，我们将为每个对象分配一个颜色
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    #点在网格中[x_min, x_max]x[y_min, y_max]。
    # 计算每个特征向量的最大值和最小值
    #X[:, 0].min()表示取矩阵X第一维度全取（冒号左边表示）（x所在的维度），第二维度取第0个（冒号右边",0"）（y所在的维度）, [x, y]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #绘制表格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    #获得测试结果
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    #将结果放入颜色图中，shape形状，reshape改造
    Z = Z.reshape(xx.shape)
    #画图，虽然从结果上看没作用
    plt.figure()
    #绘制背景颜色
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    #画出训练点,https://blog.csdn.net/qiu931110/article/details/68130199
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o',cmap=cmap_bold,edgecolor='k', s=20)

    #设置坐标轴的最大最小区间
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.xlabel(u'321')
    #plt.ylabel(u'123')
    plt.title("3-Class classification (k = %i, weights = '%s')"% (n_neighbors, weights))
    plt.show()
