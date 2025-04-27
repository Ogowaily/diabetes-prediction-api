from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.responses import JSONResponse
from .model import diabetes_model_predict

app = FastAPI()


class ModelInput(BaseModel):
    pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters: ModelInput):
    prediction = diabetes_model_predict(input_parameters)

    if prediction == 0:
        return JSONResponse(content={"prediction": "The person is not diabetic"})
    else:
        return JSONResponse(content={"prediction": "The person is diabetic"})
