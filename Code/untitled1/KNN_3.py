#随机数
import numpy as np
#绘图函数
import matplotlib.pyplot as plt
#机器学习库，其中包含数据集
from sklearn import neighbors, datasets
#颜色库
from matplotlib.colors import ListedColormap


# Create color maps #创建彩色地图
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



# 导入鸢尾花数据
# ，这里存储了其萼片和花瓣的长和宽，一共4个属性。鸢尾植物又分三类。
# iris里有两个属性iris.data，iris.target
# data里是一个矩阵，每一列代表了萼片或花瓣的长和宽，一共4列，每一行记录某个被测量的鸢尾植物，一共采样了150条记录
#target是一个数组，存储了data中每条记录属于哪一类鸢尾植物，所以数组的长度是150，数组元素的值因为共有3类鸢尾植物，所以不同值只有3个。
iris = datasets.load_iris()

#为了使数据看上去更加简单易懂，我们只取前两个特征。
#iris.data自变量矩阵
#iris.target
X = iris.data[:, :2]
y = iris.target


#分类距离
n_neighbors = 15
# 创建一个临近分类模型，需要设置分类距离和分类方式
clf = neighbors.KNeighborsClassifier(n_neighbors, 'distance')
#导入数据进行分析
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


# 网格中的步长
h = .02

# meshgrid()创建坐标矩阵，需要为其传递两个一维数字数组
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))




# 获得测试结果
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 将结果放入颜色图中，shape形状，reshape改造
Z = Z.reshape(xx.shape)
# 画图，虽然从结果上看没作用
plt.figure()
# 绘制背景颜色
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# 画出训练点
plt.scatter(X[:, 0], X[:, 1], c=y, marker='v', cmap=cmap_bold, edgecolor='k', s=20)




# 设置坐标轴的取值范围
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# 设置坐标轴的标识
plt.xlabel(u'X')
plt.ylabel(u'Y')

#生成图片展示
plt.show()