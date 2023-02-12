# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''
from datetime import date

# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os
app = Flask(__name__)

# Load the model
model = pickle.load(open('./model/salary_model.pkl','rb'))
svm_model = pickle.load(open('./model/svm_test_model.pkl','rb'))


from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify(
        greeting=["hello", "world"],
        date=date.today(),
    )

@app.route('/svm/',methods=['POST'])
def svm_predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    
    X_new = np.array(data['exp'])
    X_new = X_new.reshape(-1,2)
   
    # Make prediction using model loaded from disk as per the data.
    prediction = svm_model.predict(X_new)
    # Take the first value of prediction
    output = prediction[0]
    output = str(output)
    return jsonify(output)
    #return jsonify({'prediction': output})



@app.route('/api/',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    print (data)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])

    # Take the first value of prediction
    output = prediction[0]
   
    return jsonify(output)

if __name__ == '__main__':
    try:
        app.run(port=8000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")
