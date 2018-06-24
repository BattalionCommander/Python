import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

#导入鸢尾花数据集，并打印
iris = datasets.load_iris()
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)


#将数据存储到X和Y中，为了方便调用和观察，对数据进行重命名
x = pd.DataFrame(iris.data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']


# 设置画板的长宽
plt.figure(figsize=(7,7))
# 创建一个颜色集合
colormap = np.array(['pink', 'pink', 'pink'])
#绘图
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.xlabel(u'Petal_Length')
plt.ylabel(u'Petal_Width')
plt.title('Petal')
#展示
plt.show()