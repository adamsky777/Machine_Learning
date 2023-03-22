#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 
ML Practice on Boston Dataset
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd


X, y = load_boston(return_X_y=(True))

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor(n_neighbors=1))
    ])
    
pipe.get_params() # get the params that can be tweaked in the object.

mod = GridSearchCV(estimator=pipe, param_grid={
            'model__n_neighbors': [1,2,3,4,5,6,7,8,9,10]},
            cv=3)# we want grid search to do cross validation as well.


mod.fit(X,y)
df = pd.DataFrame(mod.cv_results_)

mod.fit(X,y)
#mod = KNeighborsRegressor() 
pred = mod.predict( X)
plt.scatter(pred,y)
plt.title("With cross validation")
#plt.xlabel("Pr") #label
