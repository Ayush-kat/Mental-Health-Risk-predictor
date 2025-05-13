import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from PIL import Image

# page config
st.set_page_config(
  page_title="Mental Health risk Predictor",
  page_icon="🧠",
  layout="wide"
)

# main title and discription
st.title("🧠 Mental Health risk Predictor")
st.subheader("Curious how your daily habits affect your mental health? let's find out together!")


image = Image.open('./src/resources/awareness.png')
col1,col2,col3 = st.columns([.05,.8,.05])
with col2:
  st.image(image)

st.subheader("Enter your lifestyle factors to get a mental health risk assessment")

#creating two columns 
col4,col5 = st.columns(2)
with col4:
  #slider input
  sleep_hours = st.slider(
     "Sleep hours ( 1-12 ) ",
     0.0,12.0,7.0,
     help = "how many hours do you sleep on average"
  )
  exercise_hours = st.slider(
     "Weekly exercise hours",
      0,10,5,
     help = "how many hours do you exercise per week"
  )
  stress_level = st.slider(
     "Stress Level ( 1-10 ) ",
     0,10,5,
     help = "Rate your stress level (1= very low, 10 = very high)"
  )
  social_activity = st.slider(
     "Social activity level ( 1-10 ) ",
     0,10,5,
     help = "Rate your social activity level (1= very low, 10 = very high)"
  )
  work_hours = st.slider(
     "Daily work hours",
     0.0,16.0,8.0,
     help = "How many hours you work daily"
  )
  screen_time = st.slider(
     "Daily Screen Time hours",
     0,10,5,
     help = "how many hours do you spend on screens (laptop,smartphone,tablet,others)?"
  )

#prediction button
if st.button("Predict Risk"):
  data = {
      "sleep_hours": sleep_hours,
      "exercise_hours": exercise_hours,
      "stress_level": stress_level,
      "social_activity":  social_activity,
      "work_hours": work_hours,
      "screen_time": screen_time 
  }
  try:
    # making post request to backend
    response = requests.post("http://localhost:8000/predict", json=data)
    if response.status_code == 200:
      prediction = response.json()['prediction']
      st.success(f"Predicted Risk level: {prediction}")
    else:
      st.error("Error making the prediction.please try again")
  except requests.exceptions.ConnectionError:
    st.error("couldn't connect the prediction service ")


# Right column history display
with col5:
  st.subheader("Recent Prediction")
  try:
    #fecting the prediction history from backend
    response = requests.get("http://localhost:8000/history")
    if response.status_code == 200:
      history = response.json()
      if history:
          df = pd.DataFrame(history)
          df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
          df['TimeStamp'] = (df['TimeStamp']).dt.strftime('%y-%m-%d %H:%M')
          # History table with selected column display
          st.dataframe(
            df[['TimeStamp','sleep_hours','exercise_hours','stress_level','social_activity',
            'work_hours','screen_time','prediction']]
          )
      else:
        st.info("no prediction history available yet!")
  except requests.exceptions.ConnectionError:
      st.error("couldn't connect the prediction service ") 









