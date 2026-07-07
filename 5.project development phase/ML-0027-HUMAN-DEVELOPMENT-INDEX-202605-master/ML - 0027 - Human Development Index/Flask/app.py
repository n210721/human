# importing the necessary dependencies
import numpy as np  # used for numerical analysis
import pandas as pd  # used for data manipulation
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)  # initializing a flask app

# loading the model - use absolute path relative to this script
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HDI.pkl')
model = pickle.load(open(model_path, 'rb'))  # loading the model


@app.route('/')  # route to display the home page
def home():
    return render_template('home.html')  # rendering the home page


@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('indexnew.html')


@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def predict():
    try:
        # reading the inputs given by the user
        life_expectancy = float(request.form['Life expectancy at birth'])
        expected_schooling = float(request.form['Expected years of schooling'])
        mean_schooling = float(request.form['Mean years of schooling'])
        gni = float(request.form['Gross national income per capita'])

        # Create DataFrame with exact feature names the model was trained on
        features_name = [
            'Life expectancy at birth',
            'Expected years of schooling',
            'Mean years of schooling',
            'Gross national income per capita'
        ]
        features_value = [[life_expectancy, expected_schooling, mean_schooling, gni]]
        df = pd.DataFrame(features_value, columns=features_name)

        # predictions using the loaded model file
        output = model.predict(df)
        y_pred = round(output[0], 4)

        # Determine HDI category
        if y_pred < 0.550:
            category = 'Low'
            color = '#ef4444'
            icon = '🔴'
        elif y_pred < 0.700:
            category = 'Medium'
            color = '#f59e0b'
            icon = '🟡'
        elif y_pred < 0.800:
            category = 'High'
            color = '#3b82f6'
            icon = '🔵'
        else:
            category = 'Very High'
            color = '#10b981'
            icon = '🟢'

        return render_template(
            "resultnew.html",
            prediction_value=y_pred,
            prediction_category=category,
            category_color=color,
            category_icon=icon,
            life_exp=life_expectancy,
            exp_school=expected_schooling,
            mean_school=mean_schooling,
            gni_val=gni
        )

    except Exception as e:
        return render_template(
            "resultnew.html",
            prediction_value=None,
            prediction_category='Error',
            category_color='#ef4444',
            category_icon='⚠️',
            error_message=str(e)
        )


if __name__ == '__main__':
    # running the app
    app.run(debug=True, port=5000)
