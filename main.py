import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load data and model
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    try:
        locations = sorted(data['location'].unique())
        print(f"Locations retrieved: {locations}")  # Debug print
    except Exception as e:
        print(f"Error retrieving locations: {e}")
        locations = []
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    try:
        print(f"Location: {location}, BHK: {bhk}, Bath: {bath}, Sqft: {sqft}")
    except Exception as e:
        print(f"Error printing values: {e}")

    # Prepare input for prediction
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    try:
        prediction = pipe.predict(input_df)[0] * 1e5
        return str(np.round(prediction, 2))
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error making prediction"

if __name__ == "__main__":
    app.run(debug=True, port=5001)
