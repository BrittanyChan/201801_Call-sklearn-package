'''
决策树  Decision Tree
既可以运用于类别变量（categorical variables）也可以作用于连续变量。
这个算法可以让我们把一个总体分为两个或多个群组。分组根据能够区分总体的
最重要的特征变量/自变量进行。
https://www.jianshu.com/p/59b510bafb4d
'''

#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
clf = tree.DecisionTreeClassifier() #clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(iris.data, iris.target)
# print (clf.score(iris.data, iris.target)) # 模型预测打分（预测精度）

#Predict Output
print (iris.data[:1, :])
print (clf.predict(iris.data[:1, :]))
# the probability of each class
print (clf.predict_proba(iris.data[:1, :])) # 计算属于每个类的概率