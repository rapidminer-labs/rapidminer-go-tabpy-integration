from tabpy_client import Client
import os
import sys
from rapidminer_go_python import rapidminergoclient as amw
import importlib
from pandas.io.json import json_normalize


TASK_STATE = 'state'
CRITERIA = 'performance'
SUB_CRITERIA = 'percentages'
PARAM = 'accuracy'
FINISHED_STATUS = 'FINISHED'
ERROR_STATUS = 'ERROR'
COMPLETE_STATUS = [FINISHED_STATUS,ERROR_STATUS]
DATA_ID = 'id'
DEPLOYMENT_ID = 'DeploymentID'
STATUS = 'Deployment_Status'
MODEL = 'Deployed_Model'
depID = ''

client = ''

tabclient = Client('http://localhost:9004/')
#new update
def connection(go_url, gouser, gopassword):
    # To get the AMW instance
    global client
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)

def rapidminer_quick_autommodel(go_url, gouser, gopassword, input_data, label):
    from rapidminer_go_python import rapidminergoclient as amw
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)
    data = json_normalize(input_data)
    return client.quick_automodel(data,label)

def rapidminerTrain(go_url, gouser, gopassword, input_data, label, platform):
    LABEL_ATTRIBUTE = label
   # To refresh the changes made in AutoModelWeb
    from rapidminer_go_python import rapidminergoclient as amw
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)
    # To get the AMW instance
    #connection(go_url, gouser, gopassword)


    data = json_normalize(input_data)

    # dataframe to json
    responseJSON = client.add_dataFrame(data)
    dataId = responseJSON[DATA_ID]

    modelingTask = client.create_modeling_task(dataId)
    modelingTaskID = modelingTask[DATA_ID]

    # setting label
    client.set_label(modelingTaskID, LABEL_ATTRIBUTE)
    client.set_class_interest(modelingTaskID, 'Yes', 'No')
    jsonVal = client.set_cost_matrix(modelingTaskID, [[1, -1], [-1, 1]])
    print('TaskID:' + modelingTaskID)

    # Initiating model training
    client.start_execution(modelingTaskID)
    flag = True
    print('ExecutingModel...')
    # Loops till all the task are completed
    while (flag):
        flag = False
        r = client.get_modeling_execution(modelingTaskID)
        states = map(lambda x: x[TASK_STATE], r)
        for state in states:
            if (str(state).strip() not in COMPLETE_STATUS):
                flag = True

    print('Model Execution Completed')

    # Obtaining the trained model results
    result = client.get_execution_result(modelingTaskID)

    # To find the best model**add or remove features if needed to get a value of more or less deep rooted in Json*
    features = [CRITERIA, SUB_CRITERIA, PARAM]
    bestModel = client.determine_best_model(features)

    # Deploying the best model
    global depID
    depID = client.deploy_model(modelingTaskID, bestModel)
    status = 'Failed'
    if str(depID) != '':
        status = 'Success'

    # Binding DeploymentID, Status and Best Model together in a dictionary to return as a output
    out_result = {DEPLOYMENT_ID: depID, STATUS: status, MODEL: bestModel}
    final_out = json_normalize(out_result)
    print('DeploymentID:' + str(depID))

    if platform == 'tabprep':
        return out_result

   # if platform == 'interactive':
   #     modelInputs = jsonVal['modelInputs']

        # To find the best model**add or remove features if needed to get a value of more or less deep rooted in Json*
   #     features = [CRITERIA, SUB_CRITERIA]
   #     jsontoDf = client.resultPerformancePercentageView(features)
   #     out_result = ({'Input:': modelInputs, 'FinalMetrics':jsontoDf})
   #     return out_result

    prediction = []

    # Number of records in input
    max_length = len(data.index)
    max_data_length = len(data.index)

    # Adding survived to a list
    for i in range(0, max_data_length):
        prediction.append(data.iloc[i][label])

    print('returning result')
    return prediction

def rapidminerScore(go_url, gouser, gopassword, jsonData, label, depID):
    print("URL is " + go_url)
    global client
    from rapidminer_go_python import rapidminergoclient as amw
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)

    print('Inside Score Method, DeploymentID'+str(depID))
    PREDICTION = 'prediction(' + label + ')'


    # passing the test data to deployed model to score
    scoreResult = client.score(jsonData, depID)

    # converting result json to dataframe
    result = json_normalize(scoreResult['data'])
    req = json_normalize(jsonData)
    # req.is_copy = True

    # List to add the result data
    prediction = []

    # Number of records in input
    max_length = len(req.index)

    # Adding confidence and predictions to a list
    for i in range(0, max_length):
        prediction.append(result.iloc[i][PREDICTION])

    print('Completed Successfully')
    return prediction



def rapidminerTrainAndScore(gouser, gopassword, dataId, jsonData, label, platform):
    # List to add the result data
    from rapidminer_go_python import rapidminergoclient as amw
    prediction = []

    prediction.extend(rapidminerTrain(gouser, gopassword, dataId, label, platform))

    prediction.extend(rapidminerScore(gouser, gopassword, jsonData, label, depID))
    print('returning result')
    return prediction



print('Deploying Quick Model')
tabclient.deploy('rapidminer_quick_autommodel',
              rapidminer_quick_autommodel,
              'Trains a model for  predictions', override=True)

print('Deploying Training and Score')
tabclient.deploy('RapidMinerTrainAndScore',
              rapidminerTrainAndScore,
              'Trains and Returns a dataset with predictions', override=True)


print('Deploying Score')
tabclient.deploy('RapidMinerScore',
              rapidminerScore,
              'Returns a dataset with predictions', override=True)



print('Deploying Training Function ')
tabclient.deploy('RapidMinerTrain',
              rapidminerTrain,
              'Trains and returns the id of deployed best model', override=True)
