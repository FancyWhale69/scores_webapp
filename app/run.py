import json
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from flask import Flask
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# load model
FILE_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR2 = os.path.join(FILE_DIR, 'models') 
model = joblib.load(PARENT_DIR2+'model.pkl')

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
