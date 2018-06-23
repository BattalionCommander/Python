import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
#颜色库
from matplotlib.colors import ListedColormap

# 创建颜色集
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# 导入鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 提取数据并处理，构建坐标矩阵，注意，y_min和y_max存储的是花萼的宽度，不是鸢尾花的种类
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#weights 平均权重，uniform一般平均，  distance与距离成反比
for weights in ['uniform', 'distance']:
    # 创建一个KNN分类器
    k = 5
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights=weights)
    #用训练数据拟合分类器
    clf.fit(X, y)


    #ravel()函数将xx和yy的两个矩阵转变成一维数组，由于两个矩阵大小相等，因此两个一维数组大小也相等
    #np.c_[]将xx.ravel()得到的列和yy.ravel()得到的列转换成矩阵
    #clf.predict()用训练好的分类器去预测结果
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # shape函数返回一个整型数字的元组，包含xx的行和列的数量
    #reshape()函数将Z的测试结果转换为两个特征数据（长度和宽度），这里需要shape函数为其传递矩阵的行和列的数量
    Z = Z.reshape(xx.shape)
    # 创建画板
    plt.figure()
    #pcolormesh函数将xx,yy两个网格矩阵和对应的预测结果Z绘制在图片上
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # 画出训练点
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap=cmap_bold, edgecolor='k', s=40)


    # 设置坐标轴的取值范围
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 设置坐标轴的标识
    plt.xlabel(u'Sepal_Length')
    plt.ylabel(u'Sepal_Width')
    plt.title("3-Class classification (k = %i, weights = '%s')" % (k, weights))
    #生成图片展示
    plt.show()
