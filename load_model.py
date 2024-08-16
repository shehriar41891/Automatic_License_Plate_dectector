import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ROBO_API_KEY')

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("number-plate-detection-mtpk0")
model = project.version('1').model
