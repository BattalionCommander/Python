import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np


iris = datasets.load_iris()
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)


#将数据存储到Pandas库中的DataFrame容器 二维，Series容器，最常用
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']


# 设置画板的长宽
plt.figure(figsize=(14,7))
# 创建一个颜色集合
colormap = np.array(['red', 'pink', 'gainsboro'])

# 为花萼创建一个子图
#subplot(numRows, numCols, plotNum)
#图表的整个绘图区域被分成 numRows 行和 numCols 列
#然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1
#plotNum 参数指定创建的 Axes 对象所在的区域

plt.subplot(1, 2, 1)
plt.scatter(x.Sepal_Length, x.Sepal_Width, c=colormap[y.Targets], s=40)
plt.title('Sepal')

# 为花瓣创建一个子图
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Petal')

plt.show()