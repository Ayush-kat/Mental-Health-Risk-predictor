
# importing all the necessary utilities we need to define the database model(table) using 
# sqlalchemy like (create_engine, declerative_base, sessionmaker)
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine

# can be defined in one line like (from sqlalchemy.orm import declarative_base,sessionmaker)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# define the database url where sqlalchemy can find the .db file or if such file doesn't 
# exist then create that file


engine = create_engine('sqlite:///health_prediction.db',connect_args={'check_same_thread': False})

#create the base class for declarative models
Base = declarative_base()

#defining our table using utilies declarative_base
class HealthData(Base):
    __tablename__ = "health-data"
    # Primary key to identify each row uniquely
    id = Column(Integer,primary_key=True,index=True)
    # user input
    sleep_hours = Column(Float)
    exercise_hours = Column(Float)
    stress_level = Column(Integer)
    social_activity = Column(Integer)
    work_hours = Column(Float)
    screen_time = Column(Float)

    # model prediction output
    prediction = Column(String)
    
    TimeStamp = Column(DateTime,default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# creating session factory that creates session(temperory interactive connection) 
SessionLocal = sessionmaker(bind=engine)

  # making a function to create a session name get_db() 
  # works as a helper or generator function that produce session and give it to api or other 
  # things those need and after the job done it runs the finally block and close the session

  # Think it like you borrow the bike for ride and when you done 
  # you return the bike (if we use return then that means we borrow the bike but never return it)

def get_db():
  session = SessionLocal()
  try:
     yield session
  finally:
     session.close()



