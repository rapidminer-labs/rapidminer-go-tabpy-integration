import tabpy
import json
from tabpy_client import Client
import pandas as pd

tabclient = Client('http://localhost:9004/')
go_url = 'https://go-develop.rapidminer.com'
go_username = 'vsivakumar@rapidminer.com'
go_password =  'wXv2jBF&v2Nm'


#values to be changed based on data
depID = '878f901c-757f-48bb-935e-36d643d6b40c'
label = 'Survived'
PREDICTION = 'prediction('+label+')'

def score(test):

    #removing rows with label values
    test = test[pd.isnull(test[label])]
    # dataframe to json
    jd = test.to_json(orient='records')
    jsonData = json.loads(jd)
    returnResult = tabclient.query('RapidMinerScore',go_url, go_username , go_password, jsonData, label, depID)
    test[PREDICTION] = returnResult['response']
    return test

#This function defines the output schema
#***Change the schema according to your result***
def get_output_schema():
  return pd.DataFrame({
    'Row No.' : prep_decimal(),
    'Age': prep_decimal(),
    'Passenger': prep_string(),
    'Sex': prep_string(),
    'Siblings' : prep_decimal(),
    'Parents' : prep_decimal(),
    'Fair' : prep_decimal(),
    PREDICTION: prep_string()
  })