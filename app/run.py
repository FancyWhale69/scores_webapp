import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from flask import Flask
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# load model
model = joblib.load('../models/model.pkl')

@app.route('/')
def index():
        
    # render web page 
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction


    return render_template('index.html', prediction_text='Post-test score: {}'.format(np.round(prediction[0], 2)))

def main():
    app.run(port=5000, debug=True)

if __name__ == '__main__':
    main()
