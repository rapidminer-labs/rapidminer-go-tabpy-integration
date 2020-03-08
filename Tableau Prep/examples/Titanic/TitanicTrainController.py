import tabpy
import json
from tabpy_client import Client
from pandas.io.json import json_normalize
import pandas as pd

# Change the following values
tabpy_serverurl = 'http://localhost:9004/'

go_url = 'https://go.rapidminer.com'
go_username = 'username here'
go_password =  'password here'

#values to be changed based on data
label = '<your label column name here'
tabclient = Client(tabpy_serverurl)


STATUS = 'Deployment_Status'
MODEL = 'Deployed_Model'
DEPLOYMENT_ID = 'DeploymentID'


def training(data):

    # removing rows with label values
    data = data.dropna(subset=[label])
    tabclient = Client(tabpy_serverurl)

    # dataframe to json+
    responseJSON = data.to_json(orient='records')
    dataId = json.loads(responseJSON)
    returnResult = tabclient.query('RapidMinerTrain', go_url, go_username, go_password, dataId, label, 'tabprep')
    final_out = json_normalize(returnResult['response'])
    return final_out

#This function defines the output schema
#***Change the schema according to your result***
def get_output_schema():
  return pd.DataFrame({
    STATUS: prep_string(),
    DEPLOYMENT_ID: prep_string(),
    MODEL: prep_string()
  })