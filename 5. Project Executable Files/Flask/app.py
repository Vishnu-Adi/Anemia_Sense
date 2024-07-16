import numpy as np
import pickle
import pandas as pd
import warnings
from flask import Flask, request, render_template

from sklearn.tree import DecisionTreeClassifier
app = Flask(__name__)
with open('model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Log all input values
    print("Received form data:")
    for key, value in request.form.items():
        print(f"{key}: {value}")

    Gender = float(request.form["Gender"])
    Hemoglobin = float(request.form["Hemoglobin"])
    MCH = float(request.form["MCH"])
    MCHC = float(request.form["MCHC"])
    MCV = float(request.form["MCV"])
    
    features_values = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])
    print("Input features:")
    print(features_values)

    # Log the model type and shape of input
    print(f"Model type: {type(model)}")
    print(f"Input shape: {features_values.shape}")

    prediction = model.predict(features_values)
    print("Model prediction:")
    print(prediction)

    result = prediction[0]
    
    if result == 0:
        result_text = "You don't have any Anemic Disease"
    elif result == 1:
        result_text = "You have Anemic Disease"
    else:
        result_text = f"Unexpected prediction value: {result}"
        
    text = "Hence, based on Calculations: "
    return render_template("predict.html", prediction_text=text + result_text)

if __name__ == '__main__':
    app.run(debug=True)
