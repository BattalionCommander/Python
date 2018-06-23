import numpy as np
import matplotlib.pyplot as plt
#机器学习库，数据集
from sklearn import neighbors, datasets

# 导入鸢尾花数据,这里存储了其萼片和花瓣的长和宽，一共4个属性。
#其中，鸢尾花又分为三类。
# iris里有两个属性iris.data，iris.target
# data里是一个矩阵，每一列代表了萼片或花瓣的长或宽，一共4列，每一行记录某个被测量的鸢尾植物，一共采样了150条记录
#target是一个数组，存储了data中每条记录属于哪一类鸢尾植物，长度是150，因为共有3类鸢尾植物，所以数组元素的值为0,1,2。
iris = datasets.load_iris()

#为了使数据看上去更加简单易懂，我们只取花萼的长和宽。
X = iris.data[:, :2]
y = iris.target

#获得数据集中的最大值和最小值，并放大绝对值，用来设置坐标系，注意，y_min和y_max存储的是花萼的宽度，不是鸢尾花的种类
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 网格中的步长
h = .02

# meshgrid()创建坐标矩阵，需要为其传递两个一维数字数组
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


#实例点个数
k = 15
# 创建一个KNN分类器，需要设置k值和投票方式
clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
#用训练数据拟合分类器
clf.fit(X, y)


# 设置坐标轴的取值范围
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# 设置坐标轴的标识
#plt.xlabel(u'Sepal_Length')
#plt.ylabel(u'Sepal_Width')

#生成图片展示
plt.show()
