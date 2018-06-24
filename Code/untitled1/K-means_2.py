import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

#导入鸢尾花数据集，存储数据并重命名
iris = datasets.load_iris()
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
colormap1 = np.array(['pink', 'pink', 'pink'])
colormap2 = np.array(['red', 'pink', 'gainsboro'])



# 为花瓣创建一个子图
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap1[y.Targets], s=40)
plt.xlabel(u'Petal_Length')
plt.ylabel(u'Petal_Width')
plt.title('Real Classification')

# 绘制模型分类
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap2[model.labels_], s=40)
plt.xlabel(u'Petal_Length')
plt.ylabel(u'Petal_Width')
plt.title('K Mean Classification')

plt.show()