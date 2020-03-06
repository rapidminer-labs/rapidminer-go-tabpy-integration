import tabpy
import json
from tabpy_client import Client
from pandas.io.json import json_normalize
import pandas as pd

# Change the following values
tabpy_serverurl = 'http://localhost:9004/'

go_url = 'https://go-develop.rapidminer.com'
go_username = 'vsivakumar@rapidminer.com'
go_password =  'wXv2jBF&v2Nm'

#values to be changed based on data
label = 'Survived'
tabclient = Client(tabpy_serverurl)


STATUS = 'Deployment_Status'
MODEL = 'Deployed_Model'
DEPLOYMENT_ID = 'DeploymentID'


def quick_training(data):

    # removing rows with label values
    data = data.dropna(subset=[label])
    tabclient = Client(tabpy_serverurl)

    # dataframe to json+
    responseJSON = data.to_json(orient='records')
    input_data = json.loads(responseJSON)
    returnResult = tabclient.query('rapidminer_quick_autommodel', go_url, go_username, go_password, input_data, label)
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