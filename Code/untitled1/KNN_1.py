#随机数
#绘图函数
import numpy as np
import matplotlib.pyplot as plt


#arange()参数为：起始值，末尾值，间隔大小
#它返回给定范围内的连续值，默认起始值从0开始，返回值中不包括末尾值
a = np.arange(1, 10, 0.1)
b = np.arange(1, 10, 1)
c = np.arange(1, 10, 2)


# meshgrid()创建坐标矩阵，需要为其传递两个一维数字数组
xx, yy = np.meshgrid(np.arange(0,10, 1), np.arange(0, 10,1))


# 设置坐标轴的取值范围
plt.xlim(xx.min(), yy.max())
plt.ylim(yy.min(), yy.max())

# 设置坐标轴的标识
plt.xlabel(u'Sepal_Length')
plt.ylabel(u'Sepal_Width')

#生成图片展示
plt.show()