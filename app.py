from flask import Flask, render_template, request, send_file, flash , redirect , url_for ,  make_response
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import io
from fpdf import FPDF
import os
from io import BytesIO

# Specify the location of the templates folder
app = Flask(__name__, template_folder='Static/templates', static_folder='Static')
app.secret_key = os.getenv('APP_SECRET_KEY')


# Load Datasets
data_symptoms = pd.read_csv("Datasets/symtoms_df.csv")
data_medication = pd.read_csv("Datasets/medications.csv")
data_diets = pd.read_csv("Datasets/diets.csv")
data_description = pd.read_csv("Datasets/description.csv")
data_precaution = pd.read_csv("Datasets/precautions_df.csv")
precautions = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
data_workout = pd.read_csv("Datasets/workout_df.csv")
df = pd.read_csv("Datasets/Training.csv")
# Load Model
svc_production = pickle.load(open('Models/production_model.pkl', 'rb'))


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Symptom Categories
symptom_categories = {
    'Skin': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'red_spots_over_body',
        'dischromic _patches', 'pus_filled_pimples', 'blackheads', 'scurring',
        'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
        'inflammatory_nails', 'blister', 'red_sore_around_nose',
        'yellow_crust_ooze'
    ],
    'Respiratory System': [
        'continuous_sneezing', 'shivering', 'cough', 'breathlessness',
        'runny_nose', 'congestion', 'sinus_pressure', 'phlegm',
        'throat_irritation', 'redness_of_eyes', 'watering_from_eyes',
        'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum'
    ],
    'Urinary System': [
        'burning_micturition', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'yellow_urine', 'dark_urine',
        'spotting_ urination', 'passage_of_gases'
    ],
    'Digestive System': [
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting',
        'abdominal_pain', 'diarrhoea', 'constipation', 'indigestion',
        'loss_of_appetite', 'nausea', 'belly_pain', 'stomach_bleeding',
        'distention_of_abdomen'
    ],
    'Musculoskeletal System': [
        'joint_pain', 'back_pain', 'neck_pain', 'cramps', 'knee_pain',
        'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
        'movement_stiffness', 'muscle_wasting', 'muscle_pain'
    ],
    'Nervous System': [
        'headache', 'dizziness', 'altered_sensorium', 'depression',
        'irritability', 'lack_of_concentration', 'visual_disturbances',
        'slurred_speech', 'spinning_movements', 'loss_of_balance',
        'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell'
    ],
    'Cardiovascular System': [
        'fast_heart_rate', 'palpitations', 'chest_pain', 'swollen_legs',
        'swollen_blood_vessels', 'puffy_face_and_eyes', 'prominent_veins_on_calf'
    ],
    'Endocrine System': [
        'weight_gain', 'weight_loss', 'cold_hands_and_feets', 'mood_swings',
        'lethargy', 'irregular_sugar_level', 'enlarged_thyroid',
        'brittle_nails', 'swollen_extremeties', 'increased_appetite',
        'polyuria', 'excessive_hunger'
    ],
    'Reproductive System': [
        'abnormal_menstruation', 'dischromic _patches', 'extra_marital_contacts'
    ],
    'General Symptoms': [
        'fatigue', 'malaise', 'weakness_in_limbs', 'high_fever', 'mild_fever',
        'sweating', 'dehydration', 'sunken_eyes', 'toxic_look_(typhos)',
        'irritability'
    ]
}

@app.context_processor
def inject_symptom_categories():
    return dict(symptoms_by_body_part=symptom_categories)

# Example function that processes selected symptoms
def get_symptom_category(symptom):
    for category, symptoms in symptom_categories.items():
        if symptom in symptoms:
            return category
    return None  # If the symptom is not found

selected_symptom = 'lack_of_concentration'  # Example symptom
category = get_symptom_category(selected_symptom)


if category is None:
    print(f"Error: '{selected_symptom}' is not a valid symptom.")
else:
    print(f"The symptom '{selected_symptom}' belongs to the category '{category}'.")
# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc_production.predict([input_vector])[0]]

# Method prediction for individual diagnostic
def get_everything(predicted_disease):
    # Initialization of lists
    user_disease_description_list = None
    user_disease_precaution_list = []
    user_disease_medication_list = None
    user_disease_diets_list = None
    user_disease_workout_list = None

    # Retrieve the description
    if 'Disease' in data_description.columns and 'Description' in data_description.columns:
        if predicted_disease in data_description['Disease'].values:
            user_disease_description = data_description.loc[data_description['Disease'] == predicted_disease, 'Description'].values
            if user_disease_description.size > 0:
                user_disease_description_list = user_disease_description[0].strip(".")

    # Retrieve the precautions
    precaution_columns = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    if 'Disease' in data_precaution.columns:
        if predicted_disease in data_precaution['Disease'].values:
            precautions = data_precaution.loc[data_precaution['Disease'] == predicted_disease, precaution_columns].values
            for row in precautions:
                user_disease_precaution_list.extend([pre for pre in row if pd.notna(pre)])

    # Retrieve the medication
    if 'Disease' in data_medication.columns and 'Medication' in data_medication.columns:
        if predicted_disease in data_medication['Disease'].values:
            user_disease_medication = data_medication.loc[data_medication['Disease'] == predicted_disease, 'Medication'].values
            if user_disease_medication.size > 0:
                user_disease_medication_list = user_disease_medication[0].strip(".")

    # Retrieve the diet
    if 'Disease' in data_diets.columns and 'Diet' in data_diets.columns:
        if predicted_disease in data_diets['Disease'].values:
            user_disease_diets = data_diets.loc[data_diets['Disease'] == predicted_disease, 'Diet'].values
            if user_disease_diets.size > 0:
                diet_value = user_disease_diets[0]
                if isinstance(diet_value, str):
                    user_disease_diets_list = eval(diet_value) if not pd.isna(diet_value) else []
                else:
                    user_disease_diets_list = [food for food in diet_value if not pd.isna(food)]

    # Retrieve the workout
    if 'disease' in data_workout.columns and 'workout' in data_workout.columns:
        if predicted_disease in data_workout['disease'].values:
            user_disease_workout = data_workout.loc[data_workout['disease'] == predicted_disease, 'workout'].values
            if user_disease_workout.size > 0:
                user_disease_workout_list = [w for w in user_disease_workout if not pd.isna(w)]

    return user_disease_description_list, user_disease_precaution_list, user_disease_medication_list, user_disease_diets_list, user_disease_workout_list




@app.route('/')
def home():
    return render_template('Home_page.html')


@app.route('/curetrack')
def curetrack():
    return render_template('CureTrack.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        selected_symptoms = request.form.get('symptoms')
        # if not selected_symptoms:
        #     # No symptoms selected, redirect back to the form with an error message
        #     return render_template('CureTrack.html', error="Please select one or more symptoms.")
        if not selected_symptoms:
            # If no symptoms are selected, flash a message to the user
            flash("Please select at least one symptom before submitting.", "warning")
            return redirect(url_for('curetrack')) 
        
        user_symptons = [s.strip() for s in selected_symptoms.split(",")]
        user_sympton = [sym.strip("[]' ") for sym in user_symptons]
        predicted_disease = get_predicted_value(user_sympton)
        user_disease_description, user_disease_precaution, user_disease_medication, user_disease_diets, user_disease_workout = get_everything(predicted_disease)

        return render_template('CureTrack.html', 
                               disease=predicted_disease, 
                               description=user_disease_description,
                               precautions=user_disease_precaution,
                               medication=user_disease_medication,
                               diets=user_disease_diets,
                               workout=user_disease_workout
                               )
        
# @app.context_processor
# def inject_symptom_categories():
#     return dict(symptom_categories=symptoms_by_body_part)

# PDF Generation Function
def generate_pdf(predicted_disease, des, pre, med, diet, workout):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Prescription Details", ln=True, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Disease: {predicted_disease}", ln=True)
    pdf.ln(5)
    
    pdf.cell(200, 10, txt=f"Description: {des}", ln=True)
    pdf.ln(5)
    
    pdf.cell(200, 10, txt="Precautions:", ln=True)
    for precaution in pre:
        pdf.cell(200, 10, txt=f"- {precaution.strip()}", ln=True)
    pdf.ln(5)
    
    pdf.cell(200, 10, txt=f"Medication: {med}", ln=True)
    pdf.ln(5)
    
    pdf.cell(200, 10, txt="Diets:", ln=True)
    for diet_item in diet:
        pdf.cell(200, 10, txt=f"- {diet_item.strip()}", ln=True)
    pdf.ln(5)
    
    pdf.cell(200, 10, txt="Workouts:", ln=True)
    for workout_item in workout:
        pdf.cell(200, 10, txt=f"- {workout_item.strip()}", ln=True)
    
    # Save to BytesIO object
    pdf_output = io.BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)  # Move to the beginning of the BytesIO object
    
    return pdf_output


@app.route('/download_prescription', methods=['POST'])
def download_prescription():
    try:
        predicted_disease = request.form['predicted_disease']
        des = request.form.get('disease_description', 'N/A')
        pre = request.form.get('disease_precaution', 'N/A').split(',')
        med = request.form.get('disease_medication', 'N/A')
        diet = request.form.get('disease_diet', 'N/A').split(',')
        workout = request.form.get('disease_workout', 'N/A').split(',')

        pdf_output = generate_pdf(predicted_disease, des, pre, med, diet, workout)
        
        return send_file(pdf_output, as_attachment=True, download_name="prescription.pdf", mimetype='application/pdf')
    except Exception as e:
        print(f"An error occurred: {e}")
        flash("An error occurred while generating the prescription. Please try again.", "danger")
        return redirect(url_for('curetrack'))

@app.route('/test_pdf')
def test_pdf():
    pdf_output = generate_pdf('Test Disease', 'Test Description', ['Precaution 1', 'Precaution 2'], 'Test Medication', ['Diet 1', 'Diet 2'], ['Workout 1', 'Workout 2'])
    return send_file(pdf_output, as_attachment=True, download_name="test_prescription.pdf", mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)
