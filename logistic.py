'''
逻辑回归  Logistic Regression
逻辑回归其实是一个分类算法而不是回归算法。通常是利用已知的自变量来预测
一个离散型因变量的值（像二进制值0/1，是/否，真/假）。简单来说，它就是
通过拟合一个逻辑函数来预测一个事件发生的概率。所以它预测的是一个概率值，
它的输出值应该在0到1之间。
'''

#Import Library
from sklearn.linear_model import LogisticRegression
from numpy import *
# Turn array into numpy(pre-process our raw data)
def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        print (lineArr)
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
X, Y = loadDataSet(r'testSet1.txt')
print (X, Y)
x_test, y_test = loadDataSet('testSet2.txt')
print (x_test, y_test)
# Create logistic regression object
model = LogisticRegression()
# Train the model using the training sets and check score
model.fit(X, Y)
print (model.score(X, Y)) # 模型预测打分（预测精度）
# Equation coefficient and Intercept
# print('Coefficient: \n', model.coef_)
# print('Intercept: \n', model.intercept_)
# Predict Output
predicted= model.predict(x_test)
print (predicted)