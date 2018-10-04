'''
降维算法  Dimensionality Reduction Algorithms
我们手上的数据有非常多的特征。虽然这听起来有利于建立更强大精准的模型，但
它们有时候反倒也是建模中的一大难题。怎样才能从1000或2000个变量里找到最重
要的变量呢？这种情况下降维算法及其他算法，如决策树，随机森林，PCA，因子分
析，相关矩阵，和缺省值比例等，就能帮我们解决难题。
'''

#Import Library
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X) # 先拟合数据，然后转化它将其转化为标准形式
print (reduced_X)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()