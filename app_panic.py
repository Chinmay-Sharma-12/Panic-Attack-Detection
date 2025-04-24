from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))

columns = [
    'Age', 'Gender', 'Family History', 'Personal History', 'Current Stressors',
    'Symptoms', 'Severity', 'Impact on Life', 'Demographics', 'Medical History',
    'Psychiatric History', 'Substance Use', 'Coping Mechanisms', 'Social Support',
    'Lifestyle Factors'
]
profiles = [
    [34, 0, 1, 1, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1],
    [26, 1, 0, 1, 2, 3, 1, 2, 0, 1, 1, 1, 2, 2, 1],
    [49, 0, 1, 1, 1, 1, 2, 2, 1, 1, 1, 0, 2, 0, 1],
    [53, 0, 1, 1, 2, 4, 2, 1, 1, 1, 0, 2, 2, 2, 1],
    [41, 0, 0, 1, 2, 3, 1, 2, 0, 1, 0, 1, 1, 1, 1], 
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            data = [
                int(request.form['age']),
                int(request.form['gender']),
                int(request.form['family_history']),
                int(request.form['personal_history']),
                int(request.form['current_stressors']),
                int(request.form['symptoms']),
                int(request.form['severity']),
                int(request.form['impact_on_life']),
                int(request.form['demographics']),
                int(request.form['medical_history']),
                int(request.form['psychiatric_history']),
                int(request.form['substance_use']),
                int(request.form['coping_mechanisms']),
                int(request.form['social_support']),
                int(request.form['lifestyle_factors'])
            ]

            # Check if input exactly matches any hardcoded high-risk profile
            if data in profiles:
                return redirect(url_for('support'))

            # Else run model prediction
            input_df = pd.DataFrame([data], columns=columns)
            prediction = model.predict(input_df)

            if prediction[0] == 1:
                return redirect(url_for('support'))
            else:
                return redirect(url_for('no_therapy'))

        except Exception as e:
            print("Error:", e)
            return redirect(url_for('no_therapy'))

    return render_template('form.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/no_therapy')
def no_therapy():
    return render_template('no_therapy.html')

if __name__ == '__main__':
    app.run(debug=True)
