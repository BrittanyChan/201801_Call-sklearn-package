'''
支持向量机  SVM
给定一组训练样本，每个标记为属于两类，一个SVM训练算法建立了
一个模型，分配新的实例为一类或其他类，使其成为非概率二元线性
分类。一个SVM模型的例子，如在空间中的点，映射，使得所述不同
的类别的例子是由一个明显的差距是尽可能宽划分的表示。新的实施
例则映射到相同的空间中，并预测基于它们落在所述间隙侧上属于一个类别。
'''

#Import Library
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

#准备训练样本
X = [[1,8],[3,20],[1,15],[3,35],[5,35],[4,40],[7,80],[6,49]]
Y = [1,1,-1,-1,1,-1,-1,1]

##开始训练
clf=svm.SVC()  ##默认参数：kernel='rbf'
clf.fit(X,Y)

#print("预测...")
#res=clf.predict([[2,2]])  ##两个方括号表面传入的参数是矩阵而不是list

##根据训练出的模型绘制样本点
for i in X:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='*')
    else :
        plt.scatter(i[0],i[1],c='g',marker='*')
##显示绘图结果
plt.show()

##生成随机实验数据(15行2列)
rdm_arr=np.random.randint(1, 15, size=(15,2))
##回执实验数据点
for i in rdm_arr:
    res=clf.predict(np.array(i).reshape(1, -1))
    if res > 0:
        plt.scatter(i[0],i[1],c='r',marker='.')
    else :
        plt.scatter(i[0],i[1],c='g',marker='.')
##显示绘图结果
plt.show()