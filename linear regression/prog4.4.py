# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:03:46 2018

@author: Namrata
"""

import numpy as np
from sklearn import linear_model

from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target

#function to compute least squares estimate
def least_squares(X,y):
    m = X.shape[0]
    n = X.shape[1]    
    A = np.zeros((m,n+1))
    A[:,0] = np.ones(m)    
    A[:,1:] = X
    theta = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y))
    return theta

#function to compute ridge regression estimate
def ridge_reg(X,y,param):
    m = X.shape[0]
    n = X.shape[1]    
    A = np.zeros((m,n+1))
    A[:,0] = np.ones(m)    
    A[:,1:] = X    
    mat = np.zeros((n+1,n+1))
    np.fill_diagonal(mat,param)
    mat[0,0] = 0
    theta = np.dot(np.linalg.inv(np.dot(A.T,A)+mat),np.dot(A.T,y))
    return theta

#function to apply regression
def reg_est(X,theta):
    m = X.shape[0]
    n = X.shape[1]    
    A = np.zeros((m,n+1))
    A[:,0] = np.ones(m)    
    A[:,1:] = X
    y = np.dot(A,theta)
    return y
    
#splitting the data
Xtrain = X[0:400,:]
ytrain = y[0:400]
Xtest = X[400:,:]
ytest = y[400:]

#standardization
avg = np.mean(Xtrain,axis=0)
var = np.std(Xtrain,axis=0)
Xtrain_std = (Xtrain-avg)/var
Xtest_std = (Xtest-avg)/var

#least squares estimation
theta = least_squares(Xtrain_std,ytrain)
y_est = reg_est(Xtest_std,theta)
mse = np.mean((ytest - y_est)**2)

print 'Least Squares'
print 'MSE = %f ' %mse

#ridge regression

#finding the optimal parameter
n = 300
param_val = np.logspace(2, 3, 100)
mse = np.zeros(param_val.shape)

for i in range(param_val.size):
    theta = ridge_reg(Xtrain_std[0:n,:], ytrain[0:n], param_val[i])
    y_est = reg_est(Xtrain_std[n:,:], theta)
    mse[i] = np.mean((ytrain[n:] - y_est)**2)
    
param = param_val[np.argmin(mse)]
theta = ridge_reg(Xtrain_std, ytrain, param)
y_est = reg_est(Xtest_std, theta)
mse = np.mean((ytest - y_est)**2)

print '\nRidge Regression'
print 'lambda = %f' %param
print 'MSE = %f' %mse

#LASSO

#finding the optimal parameter
n = 275
param_val = np.logspace(-1, 1, 100000)
mse = np.zeros(param_val.shape)

for i in range(param_val.size):
    reg = linear_model.Lasso(alpha = param_val[i])
    reg.fit(Xtrain_std[0:n,:], ytrain[0:n])
    y_est = reg.predict(Xtrain_std[n:,:])
    mse[i] = np.mean((ytrain[n:] - y_est)**2)
    
param = param_val[np.argmin(mse)]

reg = linear_model.Lasso(alpha = param)
reg.fit(Xtrain_std,ytrain)
y_est = reg.predict(Xtest_std)
mse = np.mean((ytest - y_est)**2)

print '\nLASSO'
print 'alpha = %f' %param
print 'MSE = %f' %mse
print 'Number of nonzeros = %d' %np.count_nonzero(reg.coef_)