from flask import Flask, render_template, request, redirect,url_for
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# بارگذاری مدل و مقیاس‌گر
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # دریافت ورودی‌ها از فرم
        pregnancies = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        blood_pressure = int(request.form['BloodPressure'])
        skin_thickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])

        # ساخت آرایه ورودی
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                 insulin, bmi, diabetes_pedigree, age]])
        input_data_scaled = scaler.transform(input_data)

        # پیش‌بینی با مدل
        prediction = model.predict(input_data_scaled)
        result = "دیابت دارد" if prediction[0] == 1 else "دیابت ندارد"

        return redirect(url_for('result', prediction=result))
    return render_template('index.html', result=result)

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)