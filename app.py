from flask import Flask, render_template, request , render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle



# Load Datasets
data_symptons = pd.read_csv("Datasets/symtoms_df.csv")
data_medication = pd.read_csv("Datasets/medications.csv")
data_diets = pd.read_csv("Datasets/diets.csv")
data_description = pd.read_csv("Datasets/description.csv")
data_precaution = pd.read_csv("Datasets/precautions_df.csv")
precautions = ['Precaution_1', 'Precaution_2','Precaution_3','Precaution_4']
data_workout = pd.read_csv("Datasets/workout_df.csv")
df = pd.read_csv("Datasets/Training.csv")

# Data Filtering
X = df.drop('prognosis', axis = 1)
y = df['prognosis']

# Load Model
svc_production = pickle.load(open('Models/production_model.pkl', 'rb')) 

# Feature Eng- Label Encoding 
encoder = LabelEncoder()
encoder.fit(y)
Y = encoder.transform(y)


# Mapping of encoded values to original labels
label_mapping = dict(enumerate(encoder.classes_))

# Making dict for individual symptons 
symptons = list(df.columns)
symptons_dict = {}
for index, element in enumerate(symptons):
    if index == len(symptons)-1:
        break
    symptons_dict[element] = index


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptons_dict))
    for item in patient_symptoms:
        input_vector[symptons_dict[item]] = 1
    return label_mapping[svc_production.predict([input_vector])[0]]

# Method prediction for individual diagnostic
def get_everything(predicted_disease):
    user_disease_description_list = None
    user_disease_precaution_list = None
    user_disease_medication_list = None
    user_disease_diets_list = None
    user_disease_workout_list = None

    # Retrieve the description
    if predicted_disease in data_description['Disease'].values:
        user_disease_description = data_description.loc[data_description['Disease'] == predicted_disease, 'Description'].values[0]
        user_disease_description_list = user_disease_description.strip(".")
    
    # Retrieve the precautions
    if predicted_disease in data_precaution['Disease'].values:
        precautions = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']  # Adjust this list as needed
        user_disease_precaution = data_precaution.loc[data_precaution['Disease'] == predicted_disease, precautions].values[0]
        user_disease_precaution_list = [pre for pre in user_disease_precaution if pd.notna(pre)]
    
    # Retrieve the medication
    if predicted_disease in data_medication['Disease'].values:
        user_disease_medication = data_medication.loc[data_medication['Disease'] == predicted_disease, 'Medication'].values[0]
        user_disease_medication_list = user_disease_medication.strip(".")
    
    # Retrieve the diet
    if predicted_disease in data_diets['Disease'].values:
        user_disease_diets = data_diets.loc[data_diets['Disease'] == predicted_disease, 'Diet'].values[0]
        if isinstance(user_disease_diets, str):
            user_disease_diets_list = eval(user_disease_diets)
        else:
            user_disease_diets_list = [food for food in user_disease_diets if pd.notna(food)]
    
    # Retrieve the workout
    if predicted_disease in data_workout['disease'].values:
        user_disease_workout = data_workout.loc[data_workout['disease'] == predicted_disease, 'workout'].values
        user_disease_workout_list = [w for w in user_disease_workout if pd.notna(w)]
    
    return user_disease_description_list, user_disease_precaution_list, user_disease_medication_list, user_disease_diets_list, user_disease_workout_list


# Specify the location of the templates folder
app = Flask(__name__, template_folder='Static/templates', static_folder='Static')

@app.route('/')
def home():
    return render_template('Home_page.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/curetrack')
def curetrack():
    return render_template('CureTrack.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        symptons = request.form.get('symptoms')
        user_symptons = [s.strip() for s in symptons.split(",")]
        user_sympton = [sym.strip("[]' ") for sym in user_symptons]
        predicted_disease = get_predicted_value(user_sympton)
        des, pre, med, diet, workout = get_everything(predicted_disease)
        
        return render_template('curetrack.html', predicted_disease=predicted_disease,  disease_description = des, disease_medication = med, disease_diet = diet, disease_workout = workout, disease_precaution= pre)

        
        
    
if __name__ == '__main__':
    app.run(debug=True)
