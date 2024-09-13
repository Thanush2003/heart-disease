from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import os
import joblib
import pickle

app = Flask(__name__)
a = pickle.load(open("cl.pkl","rb"))

def predictfunc(input_text):
    input_text = [input_text]  # Convert to a list of lists
    prediction = a.predict(input_text)
    if prediction == 1:
        result = True
    else:
        result = False
    return result

@app.route('/')
def home():
    return render_template('a.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        Sex_male = float(request.form['Sex_male'])
        cigsPerDay = float(request.form['cigsPerDay'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        glucose = float(request.form['glucose'])
        
        input_text = [age, Sex_male, cigsPerDay, totChol, sysBP, glucose]
        result = predictfunc(input_text)
        return render_template('b.html', res=result) 
if __name__ == '__main__':
    app.run(debug=True)