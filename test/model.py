from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_iris
from regresspy.regression import Regression
from regresspy.loss import rmse

iris = load_iris()
# We will use sepal length to predict sepal width
X = iris.data[:, 0].reshape(-1, 1)
Y = iris.data[:, 1].reshape(-1, 1)




sgd = SGDRegressor(max_iter=100,learning_rate='constant',eta0=0.001)
sgd.fit(X,Y.reshape(-1))
sgd_pred=sgd.predict(X)
sgd_rmse= rmse(sgd_pred,Y.reshape(-1))
print('SGDRegressor RMSE:',str(sgd_rmse))

reg = Regression(epoch=100,learing_rate=0.0001)
reg.fit(X,Y)
reg_pred=reg.predict(x)
reg_rmse=reg.score(reg_pred,Y)
print('RMSE of our class:',str(reg_rmse))

