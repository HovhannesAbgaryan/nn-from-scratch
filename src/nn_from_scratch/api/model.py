from pydantic import BaseModel, Field
from typing import Optional


class HeartData(BaseModel):
    age: Optional[int] = Field(None, ge=0, description="Age of the patient, must be non-negative")
    trestbps: Optional[int] = Field(None, ge=0, description="Resting blood pressure in mmHg")
    chol: Optional[int] = Field(None, ge=0, description="Serum cholesterol in mg/dl")
    thalach: Optional[int] = Field(None, ge=0, description="Maximum heart rate achieved")
    oldpeak: Optional[float] = Field(None, description="ST depression induced by exercise relative to rest")
    sex: Optional[int] = Field(None, ge=0, le=1, description="Sex: 0 for female, 1 for male")
    fbs: Optional[int] = Field(None, ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    exang: Optional[int] = Field(None, ge=0, le=1, description="Exercise-induced angina (1 = yes, 0 = no)")
    cp: Optional[int] = Field(None, ge=0, le=3, description="Chest pain type (0-3)")
    restecg: Optional[int] = Field(None, ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    slope: Optional[int] = Field(None, ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: Optional[int] = Field(None, ge=0, le=2, description="Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "trestbps": 120,
                "chol": 230,
                "thalach": 150,
                "oldpeak": 1.5,
                "sex": 1,
                "fbs": 0,
                "exang": 0,
                "cp": 2,
                "restecg": 1,
                "slope": 1,
                "ca": 0,
                "thal": 2,
            }
        }
