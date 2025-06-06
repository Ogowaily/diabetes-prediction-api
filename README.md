# Diabetes Prediction API

## Overview
The **Diabetes Prediction API** is a machine learning-based API designed to predict whether a person has diabetes or not based on input health data. It uses a pre-trained model to classify input data into two categories: **diabetic** or **non-diabetic**.

This project leverages **FastAPI** for the API implementation and uses a machine learning model that was trained on a dataset to predict diabetes based on various health attributes like glucose level, BMI, age, etc.

## Project Structure
```bash
.
├── app/                      # Contains the FastAPI application code
│   └── main.py               # FastAPI app with the POST endpoint
├── client/                   # Client-side code to make API requests
│   └── diabetes_prediction_client.py  # Script to send data and get predictions
├── model/                    # Contains the pre-trained model
│   └── diabetes_model.sav    # Pre-trained machine learning model (Pickle format)
├── requirements.txt          # Required dependencies for the project
└── README.md                 # Project documentation
```

## Technology Stack
- **FastAPI**: For building the REST API.
- **Pickle**: For loading the pre-trained machine learning model.
- **Python**: Programming language.
- **Pydantic**: For data validation and input parsing.

## How to Use

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/Ogowaily/diabetes-prediction-api.git
cd diabetes-prediction-api
```

### 2. Install Dependencies
Make sure you have Python 3.7 or higher installed. Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

### 3. Run the API Locally
Once dependencies are installed, run the FastAPI app locally:
```bash
uvicorn app.main:app --reload
```
This will start the API server on `http://127.0.0.1:8000`.

### 4. Test the API
You can test the API using any HTTP client (e.g., Postman, cURL) or through Python's `requests` library.

**Example Request**:
```python
import json
import requests

url = 'http://127.0.0.1:8000/diabetes_prediction'

input_data_for_model = {
    'pregnancies': 5,
    'Glucose': 166,
    'BloodPressure': 72,
    'SkinThickness': 19,
    'Insulin': 175,
    'BMI': 25.8,
    'DiabetesPedigreeFunction': 0.587,
    'Age': 51
}

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.json())
```

**Example Response**:
```json
{
  "prediction": "The person is diabetic"
}
```

## Model Description
The machine learning model used for diabetes prediction is a classification model that has been trained on a dataset of health records. The model takes in multiple health parameters as input, such as glucose levels, BMI, age, and more, and classifies the individual as **diabetic** or **non-diabetic**.

## API Documentation
You can also explore the interactive API documentation generated by FastAPI at the following URL when the server is running:
```
http://127.0.0.1:8000/docs
```

## License
This project is open-source and available under the [MIT License](LICENSE).
```

### Notes about Project Structure:
1. **`app/`**: Contains the FastAPI code to handle the prediction requests.
2. **`client/`**: Contains client-side code that interacts with the API (for making requests).
3. **`model/`**: Stores the pre-trained model in a file like `diabetes_model.sav` (Pickle format).
4. **`requirements.txt`**: Lists all the dependencies needed to run the project.

This structure is clear and easy to follow for users and developers. You can make the necessary adjustments if you plan to add additional details or files later.

Let me know if you need any other adjustments!
