import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.svm import SVC

labels = ['fail', 'pass']
with open('./model/svm_test_model.pkl', 'rb') as f:     
        model = pickle.load(f)

x_test = np.array([
    [4, 6]
])

y_predict = model.predict(x_test)
label = labels[y_predict[0]]

print (label)

y_predict = model.predict_proba(x_test)
confidence = y_predict[0][y_predict[0].argmax()]


print(label, confidence) #