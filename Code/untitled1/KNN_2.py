#随机数
#绘图函数
import numpy as np
import matplotlib.pyplot as plt
#机器学习库，数据集
from sklearn import neighbors, datasets

#分类距离
n_neighbors = 15


# 导入一些数据
iris = datasets.load_iris()
print(iris.data.shape())
print(iris.target.shape())

# iris中文指鸢尾植物，这里存储了其萼片和花瓣的长和宽，一共4个属性。鸢尾植物又分三类。
# 与之相对，iris里有两个属性iris.data，iris.target
# data里是一个矩阵，每一列代表了萼片或花瓣的长宽，一共4列，每一列代表某个被测量的鸢尾植物，一共采样了150条记录，所以查看这个矩阵的形状iris.data.shape，
#target是一个数组，存储了data中每条记录属于哪一类鸢尾植物，所以数组的长度是150，数组元素的值因为共有3类鸢尾植物，所以不同值只有3个。



# 我们只取前两个特征。
# slicing by using a two-dim dataset
# 使用两个模糊数据集进行切片
#iris.data自变量矩阵
#iris.target
X = iris.data[:, :2]
y = iris.target


# meshgrid()创建坐标矩阵，需要为其传递两个一维数字数组
xx, yy = np.meshgrid(np.arange(0,10, 0.2), np.arange(0, 10,2.5))


# 设置坐标轴的取值范围
plt.xlim(xx.min(), yy.max())
plt.ylim(yy.min(), yy.max())

# 设置坐标轴的标识
plt.xlabel(u'X')
plt.ylabel(u'Y')

#生成图片展示
plt.show()