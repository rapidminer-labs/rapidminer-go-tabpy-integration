import tabpy
import json
from tabpy_client import Client
import pandas as pd

tabclient = Client('http://localhost:9004/')
go_url = ''
go_username = ''
go_password =  ''


#values to be changed based on data
deployment_ID = ''
label = ''
PREDICTION = 'prediction('+label+')'

def score(test):

    #removing rows with label values
    test = test[pd.isnull(test[label])]
    # dataframe to json
    jd = test.to_json(orient='records')
    jsonData = json.loads(jd)
    returnResult = tabclient.query('RapidMinerScore',go_url, go_username , go_password, jsonData, label, deployment_ID)
    test[PREDICTION] = returnResult['response']
    return test

#This function defines the output schema
#***Change the schema according to your result***
def get_output_schema():
  return pd.DataFrame({
    'col1' : prep_decimal(),
    'col2': prep_decimal(),
    'col3': prep_string(),
     PREDICTION: prep_string()
  })