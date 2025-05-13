from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np

from models import get_db, HealthData

# now initialliazing our FastAPI
app = FastAPI(title= "Mental Health Risk predictor")

class HealthDataInput(BaseModel):
    sleep_hours: float
    exercise_hours: float
    stress_level: int
    social_activity: int
    work_hours: float
    screen_time: float

# creating a dummy model
def create_dummy_model():
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier(n_estimators=100,random_state=42)
  X = np.random.rand(100,6)
  Y = np.random.choice(['Low risk','medium risk','High risk'],size = 100)
  model.fit(X,Y)
  return model

#initialize the dummy_model
model = create_dummy_model()


@app.post("/predict")
def predict_health_risk(data: HealthDataInput, db: Session=Depends(get_db)):
    try:
      #convert input data as numpy array for prediction
      input_data = np.array([[
        data.sleep_hours,
        data.exercise_hours,
        data.stress_level,
        data.social_activity,
        data.work_hours,
        data.screen_time 
      ]])
      
      #using model to make prediction
      prediction = model.predict(input_data)[0]

      #create and save the changes to database
      db_record = HealthData(
        sleep_hours= data.sleep_hours,
        exercise_hours= data.exercise_hours,
        stress_level= data.stress_level,
        social_activity= data.social_activity,
        work_hours= data.work_hours,
        screen_time= data.screen_time,
        prediction= prediction)
      
      db.add(db_record)
      db.commit()

      return {"prediction" : prediction}
    except Exception as e:
      raise HTTPException(status_code=500,detail=str(e))
    
@app.get("/history")
def get_prediction_history(db: Session= Depends(get_db)):
   records = db.query(HealthData).order_by(HealthData.TimeStamp.desc()).limit(12).all()
   return records

