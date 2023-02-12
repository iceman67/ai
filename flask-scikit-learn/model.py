## model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

dataset = pd.read_csv('data/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 777)

salary_model = LinearRegression()
salary_model.fit(X_train, y_train)

y_pred = salary_model.predict(X_test)

pickle.dump(salary_model, open('model/salary_model.pkl','wb'))


