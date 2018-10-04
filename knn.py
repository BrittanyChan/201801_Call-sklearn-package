'''
K邻近算法  KNN
这个算法既可以解决分类问题，也可以用于回归问题，但工业上用于分类的情况更多。
KNN先记录所有已知数据，再利用一个距离函数，找出已知数据中距离未知事件最近的
K组数据，最后按照这K组数据里最常见的类别预测该事件。
'''

#Import Library

from sklearn.neighbors import KNeighborsClassifier
from numpy import *

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return (group,labels)
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
X, Y = createDataSet()
x_test = [[0.3, 1.0]]
# Create KNeighbors classifier object model
knn = KNeighborsClassifier(n_neighbors=2) # default value for n_neighbors is 5
# Train the model using the training sets
knn.fit(X, Y)
#Predict Output
predicted= knn.predict(x_test)
print (predicted)