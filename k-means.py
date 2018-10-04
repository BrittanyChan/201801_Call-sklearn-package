'''
K-均值算法  K-means
首先从n个数据对象任意选择 k 个对象作为初始聚类中心；而对于所剩下其它
对象，则根据它们与这些聚类中心的相似度（距离），分别将它们分配给与其
最相似的（聚类中心所代表的）聚类；然 后再计算每个所获新聚类的聚类中心
（该聚类中所有对象的均值）；不断重复这一过程直到标准测度函数开始收敛为止。
'''

#Import Library
from sklearn.cluster import KMeans

def loadDataSet(filename):
    dataset = []
    fr = open(filename)
    for line in fr.readlines():
        element = line.strip('\n').split(',')
        dataset.append(element)
    return dataset

#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
X = loadDataSet('kmeanstrainingdata.txt')
print (X)
x_test = loadDataSet('kmeanstestingdata.txt')
print (x_test)
# Create KNeighbors classifier object model
k_means = KMeans(n_clusters=3, random_state=0)
# Train the model using the training sets and check score
k_means.fit(X)
#Predict Output
predicted= k_means.predict(x_test)
print (predicted)