#随机数
#绘图函数
import numpy as np
import matplotlib.pyplot as plt
#机器学习库，数据集
from sklearn import neighbors, datasets

#分类距离
n_neighbors = 15

# import some data to play with
# 导入一些数据
iris = datasets.load_iris()


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