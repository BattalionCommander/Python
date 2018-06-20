#随机数
#绘图函数
import numpy as np
import matplotlib.pyplot as plt

#机器学习库，数据集
from sklearn import neighbors, datasets




# 导入鸢尾花数据
# ，这里存储了其萼片和花瓣的长和宽，一共4个属性。鸢尾植物又分三类。
# iris里有两个属性iris.data，iris.target
# data里是一个矩阵，每一列代表了萼片或花瓣的长和宽，一共4列，每一行记录某个被测量的鸢尾植物，一共采样了150条记录
#target是一个数组，存储了data中每条记录属于哪一类鸢尾植物，所以数组的长度是150，数组元素的值因为共有3类鸢尾植物，所以不同值只有3个。
iris = datasets.load_iris()
data = iris.data

#为了使数据看上去更加简单易懂，我们只取前两个特征。
#iris.data自变量矩阵
#iris.target
X = iris.data[:, :2]
y = iris.target



#分类距离
k = 15
# 创建一个临近分类模型，需要设置分类距离和分类方式
clf = neighbors.KNeighborsClassifier(k, 'uniform')
#导入数据进行分析
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1




# 网格中的步长
h = .02

# meshgrid()创建坐标矩阵，需要为其传递两个一维数字数组
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))


# 设置坐标轴的取值范围
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# 设置坐标轴的标识
plt.xlabel(u'X')
plt.ylabel(u'Y')

#生成图片展示
plt.show()