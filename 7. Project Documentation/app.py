from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model, scaler, and encoders
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        own_car = request.form['own_car']
        own_realty = request.form['own_realty']
        income = float(request.form['income'])
        income_type = request.form['income_type']
        education_type = request.form['education_type']
        family_status = request.form['family_status']
        housing_type = request.form['housing_type']
        age = int(request.form['age']) * 365
        years_employed = int(request.form['years_employed']) * 365
        fam_members = int(request.form['fam_members'])

        gender_enc = encoders['gender'].transform([gender])[0]
        own_car_enc = encoders['own_car'].transform([own_car])[0]
        own_realty_enc = encoders['own_realty'].transform([own_realty])[0]
        income_type_enc = encoders['income_type'].transform([income_type])[0]
        education_type_enc = encoders['education_type'].transform([education_type])[0]
        family_status_enc = encoders['family_status'].transform([family_status])[0]
        housing_type_enc = encoders['housing_type'].transform([housing_type])[0]

        flag_work_phone = 0
        flag_phone = 0
        flag_email = 0
        cnt_adults = fam_members


        # ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','AMT_INCOME_TOTAL',
        #  'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE',
        #  'DAYS_BIRTH','DAYS_EMPLOYED','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL',
        #  'CNT_FAM_MEMBERS','CNT_ADULTS']
        features = np.array([[gender_enc, own_car_enc, own_realty_enc, income,
                          income_type_enc, education_type_enc, family_status_enc,
                          housing_type_enc, age, years_employed,
                          flag_work_phone, flag_phone, flag_email,
                          fam_members, cnt_adults]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)