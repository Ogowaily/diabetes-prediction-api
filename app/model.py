import pickle


diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))

def diabetes_model_predict(input_parameters):
    input_data = input_parameters.dict()
    input_list = [
        input_data['pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
        input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
        input_data['DiabetesPedigreeFunction'], input_data['Age']
    ]

    prediction = diabetes_model.predict([input_list])
    return prediction[0]
