from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Error: .pkl files not found. Make sure they are in the same folder as app.py")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = {
            'Product ID': [int(request.form['product_id'])],
            'Store ID': [int(request.form['store_id'])],
            'Sales Quantity': [float(request.form['sales_quantity'])],
            'Price': [float(request.form['price'])],
            'Promotions': [request.form['promotions']],
            'Seasonality Factors': [request.form['seasonality']],
            'External Factors': [request.form['external_factors']]
        }

        df_input = pd.DataFrame(data)

        prediction_index = model.predict(df_input)
        
        prediction_label = le.inverse_transform(prediction_index)
        
        result = prediction_label[0]

        return render_template('index.html', prediction_text=f'Predicted Customer Segment: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
