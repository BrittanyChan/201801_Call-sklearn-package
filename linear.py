'''
线性回归  Linear Regression
线性回归是利用连续性变量来估计实际数值（例如房价，呼叫次数和总销售额等）。
我们通过线性回归算法找出自变量和因变量间的最佳线性关系，图形上可以确定一条
最佳直线。这条最佳直线就是回归线。这个回归关系可以用Y=aX+b 表示。
'''

#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
input_variables_values_training_datasets=[[2],
                                          [1]]
target_variables_values_training_datasets=[[5,9,11],
                                           [7,10,13]]
input_variables_values_test_datasets=[[7],
                                      [10]]
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets
# Create linear regression object
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)
#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
predicted= linear.predict(x_test)
print('Predicted:\n', predicted)