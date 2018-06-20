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


# 创建K Means算法实例,n_clusters表示簇的个数，即你想聚成几类
model = KMeans(n_clusters=3)
model.fit(x)

# 设置画板的长宽
plt.figure(figsize=(14,7))

# 创建一个颜色集合
colormap = np.array(['red', 'lime', 'black'])

# 为花瓣创建一个子图
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

# 绘制模型分类
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')

plt.show()