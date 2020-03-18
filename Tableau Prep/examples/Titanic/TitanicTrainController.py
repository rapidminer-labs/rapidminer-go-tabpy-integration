import tabpy
import json
from tabpy_client import Client
from pandas.io.json import json_normalize
import pandas as pd

# Change the following values
tabpy_serverurl = 'http://localhost:9004/'
go_url = 'https://go.rapidminer.com'
go_username = ''
go_password = ''

#values to be changed based on data
label = 'Survived'
file_name = 'Titanic'
cost_matrix =[[1,-1],[-1,1]]
high_value = 'Yes'
low_value = 'No'
#### possible values,
selection_criteria = 'performance_accuracy'
max_min_crietria_selector = 'max' #or 'min'
should_deploy = True
platform = 'tabprep'

tabclient = Client(tabpy_serverurl)
STATUS = 'Deployment_Status'
MODEL = 'Deployed_Model'
DEPLOYMENT_ID = 'DeploymentID'
MODELING_ID = 'Modeling_ID'
URL = 'URL'

def training(data):
    print('inside controller')
    # removing rows with label values
    data = data.dropna(subset=[label])
    tabclient = Client(tabpy_serverurl)

    # dataframe to json
    input_data = json.loads(data.to_json(orient='records'))
    returnResult = tabclient.query('RapidMiner_Train', go_url, go_username, go_password, file_name, input_data, label,cost_matrix,high_value,low_value,selection_criteria, max_min_crietria_selector, platform)
    final_out = json_normalize(returnResult['response'])
    return final_out

#This function defines the output schema
#***Change the schema according to your result***
def get_output_schema():
  return pd.DataFrame({
    MODELING_ID: prep_string(),
    STATUS: prep_string(),
    DEPLOYMENT_ID: prep_string(),
    MODEL: prep_string(),
    URL: prep_string()
  })