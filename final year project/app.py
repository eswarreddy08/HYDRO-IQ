from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("stacking_model.pkl", "rb"))

# Features used in the model
features = ['Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'ph']

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[feature]) for feature in features]
    input_array = np.array([input_data])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    result = "Potable" if prediction[0] == 1 else "Not Potable"
    return render_template('index.html', features=features, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
