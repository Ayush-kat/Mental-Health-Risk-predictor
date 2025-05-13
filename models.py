from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# define the database url where sqlalchemy can find the .db file or if such file doesn't 
# exist then create that file

databaseURL = 'sqlite://.health_prediction.db'
engine = create_engine(databaseURL,connect_args={'check_same_connection': False})

#create the base class for declarative models
Base = declarative_base()

class HealthData(Base):
  __tablename__ = "health-data"
  # Primary key to identify each row uniquely
  id = Column(Integer,primary_key=True,index=True)
  # user input
  Sleep_hours = Column(Float)
  exercise_hours = Column(Float)
  stress_level = Column(Integer)
  social_activity = Column(Integer)
  working_hours = Column(Float)
  ScreenTime = Column(Float)

  # model prediction output
  prediction = Column(String)
  
  TimeStamp = Column(DateTime,default=datetime.utcnow)


  # creating session using sessionmaker
  SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)
  
  # making a function to create a session name get_db() 
  def get_db():
    session = SessionLocal()



