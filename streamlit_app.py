import streamlit as st

# ========================
# Page Config (MUST be first Streamlit command)
# ========================
st.set_page_config(
    page_title="Mental Health Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import os

# Database imports
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ========================
# Database Configuration
# ========================
# Use relative path for Streamlit Cloud compatibility
DB_PATH = os.path.join(os.path.dirname(__file__), 'health_prediction.db')
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={'check_same_thread': False})

Base = declarative_base()

class HealthData(Base):
    __tablename__ = "health-data"
    id = Column(Integer, primary_key=True, index=True)
    sleep_hours = Column(Float)
    exercise_hours = Column(Float)
    stress_level = Column(Integer)
    social_activity = Column(Integer)
    work_hours = Column(Float)
    screen_time = Column(Float)
    prediction = Column(String)
    TimeStamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# ========================
# ML Model
# ========================
@st.cache_resource
def create_model():
    """Create and cache the ML model"""
    from sklearn.ensemble import RandomForestClassifier
    
    np.random.seed(42)
    n_samples = 200

    sleep_hours = np.random.uniform(4, 9, size=n_samples)
    stress_level = np.random.randint(1, 10, size=n_samples)
    social_activity = np.random.randint(0, 7, size=n_samples)
    work_hours = np.random.uniform(4, 12, size=n_samples)
    screen_time = np.random.uniform(2, 10, size=n_samples)
    physical_activity = np.random.uniform(0, 7, size=n_samples)

    X = np.column_stack([
        sleep_hours,
        stress_level,
        social_activity,
        work_hours,
        screen_time,
        physical_activity
    ])

    # Create scores
    score_values = (
        (9 - sleep_hours) +         
        stress_level * 1.5 +        
        screen_time +              
        (12 - work_hours) * 0.5 -  
        physical_activity -        
        social_activity
    )

    # Divide by percentiles
    low_th = np.percentile(score_values, 33)
    high_th = np.percentile(score_values, 66)

    Y = []
    for score in score_values:
        if score < low_th:
            Y.append('Low risk')
        elif score < high_th:
            Y.append('medium risk')
        else:
            Y.append('High risk')

    model = RandomForestClassifier(n_estimators=100, random_state=69)
    model.fit(X, Y)

    return model

# Initialize model
model = create_model()

# ========================
# Streamlit UI
# ========================

# Main title and description
st.title("🧠 Mental Health Risk Predictor")
st.subheader("Curious how your daily habits affect your mental health? Let's find out together!")

# Display image
image_path = os.path.join(os.path.dirname(__file__), 'src', 'resources', 'awareness.png')
if os.path.exists(image_path):
    image = Image.open(image_path)
    col1, col2, col3 = st.columns([0.05, 0.8, 0.05])
    with col2:
        st.image(image)

st.subheader("Enter your lifestyle factors to get a mental health risk assessment")

# Creating two columns
col4, col5 = st.columns(2)

with col4:
    # Slider inputs
    sleep_hours = st.slider(
        "Sleep hours (1-12)",
        0.0, 12.0, 7.0,
        help="How many hours do you sleep on average"
    )
    exercise_hours = st.slider(
        "Weekly exercise hours",
        0, 10, 5,
        help="How many hours do you exercise per week"
    )
    stress_level = st.slider(
        "Stress Level (1-10)",
        0, 10, 5,
        help="Rate your stress level (1= very low, 10 = very high)"
    )
    social_activity = st.slider(
        "Social activity level (1-10)",
        0, 10, 5,
        help="Rate your social activity level (1= very low, 10 = very high)"
    )
    work_hours = st.slider(
        "Daily work hours",
        0.0, 16.0, 8.0,
        help="How many hours you work daily"
    )
    screen_time = st.slider(
        "Daily Screen Time hours",
        0, 10, 5,
        help="How many hours do you spend on screens?"
    )

    # Prediction button
    if st.button("🔮 Predict Risk", type="primary"):
        # Prepare input data
        input_data = np.array([[
            sleep_hours,
            exercise_hours,
            stress_level,
            social_activity,
            work_hours,
            screen_time
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result with appropriate styling
        if prediction == 'Low risk':
            st.success(f"✅ Predicted Risk Level: **{prediction}**")
        elif prediction == 'medium risk':
            st.warning(f"⚠️ Predicted Risk Level: **{prediction}**")
        else:
            st.error(f"🚨 Predicted Risk Level: **{prediction}**")
        
        # Save to database
        try:
            db = next(get_db())
            db_record = HealthData(
                sleep_hours=sleep_hours,
                exercise_hours=exercise_hours,
                stress_level=stress_level,
                social_activity=social_activity,
                work_hours=work_hours,
                screen_time=screen_time,
                prediction=prediction
            )
            db.add(db_record)
            db.commit()
            db.close()
        except Exception as e:
            st.warning(f"Note: Could not save to history ({e})")

# Right column - history display
with col5:
    st.subheader("📊 Recent Predictions")
    try:
        db = next(get_db())
        records = db.query(HealthData).order_by(HealthData.TimeStamp.desc()).limit(12).all()
        db.close()
        
        if records:
            data = [{
                'Time': r.TimeStamp.strftime('%y-%m-%d %H:%M') if r.TimeStamp else 'N/A',
                'Sleep': r.sleep_hours,
                'Exercise': r.exercise_hours,
                'Stress': r.stress_level,
                'Social': r.social_activity,
                'Work': r.work_hours,
                'Screen': r.screen_time,
                'Risk': r.prediction
            } for r in records]
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No prediction history available yet! Make your first prediction.")
    except Exception as e:
        st.info("No prediction history available yet!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
