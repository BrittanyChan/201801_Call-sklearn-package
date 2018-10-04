'''
朴素贝叶斯  Naive Bayes
朴素贝叶斯的思想基础是这样的：对于给出的待分类项，求解在此项出现的
条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。
'''

#测试数据
import numpy as np
features_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels_train = np.array([1, 1, 1, 2, 2, 2])
#引入高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
#实例化
clf = GaussianNB()
#训练数据 fit相当于train
clf.fit(features_train, labels_train)
#输出单个预测结果
features_test = np.array([[-0.8,-1],[2, 2]])
labels_test = np.array([[1],[1]])
pred = clf.predict(features_test)
print(pred)
#准确度评估 评估正确/总数
#方法1
accuracy = clf.score(features_test, labels_test)
print (accuracy)
#方法2
from sklearn.metrics import accuracy_score
accuracy2 = accuracy_score(pred,labels_test)
print (accuracy2)