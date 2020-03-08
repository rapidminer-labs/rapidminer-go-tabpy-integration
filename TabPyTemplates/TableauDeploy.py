from tabpy_client import Client
import importlib

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
AUTODEPLOY = True
client = ''

tabclient = Client('http://localhost:9004/')
#new update

def rapidminer_quick_training(go_url, gouser, gopassword, input_data, label, platform):
    from rapidminer_go_python import rapidminergoclient as amw
    LABEL_ATTRIBUTE = label

    # To get the AMW instance
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)

    data = client.convert_json_to_dataframe(input_data)
    trainingResult = client.quick_automodel(data,label,AUTODEPLOY)

    if platform == 'tabprep':
        return trainingResult

    prediction = []

    # Number of records in input
    max_data_length = len(data.index)

    # Adding survived to a list
    for i in range(0, max_data_length):
        prediction.append(data.iloc[i][label])

    print('returning result')
    return prediction

def rapidminer_train(go_url, gouser, gopassword, input_data, label,cost_matrix,high_value,low_value,selection_criteria,should_depoly, platform):

    from rapidminer_go_python import rapidminergoclient as amw
    LABEL_ATTRIBUTE = label
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)
    data = client.convert_json_to_dataframe(input_data)
    dataId = client.add_dataFrame(data)[DATA_ID]
    modelingTaskID = client.create_modeling_task(dataId)[DATA_ID]
    # setting label
    client.set_label(modelingTaskID, LABEL_ATTRIBUTE)
    client.set_class_interest(modelingTaskID, high_value, low_value)
    jsonVal = client.set_cost_matrix(modelingTaskID, cost_matrix)
    print('TaskID:' + modelingTaskID)

    # Initiating model training
    client.start_execution(modelingTaskID)
    print('ExecutingModel...')
    # Obtaining the trained model results
    client.get_execution_result(modelingTaskID)

    # To find the best model**add or remove features if needed to get a value of more or less deep rooted in Json*
    bestmodelselectioncriteria = [CRITERIA, SUB_CRITERIA, PARAM]
    bestModel = client.determine_best_model(bestmodelselectioncriteria)

    # Deploying the best model
    global depID
    depID = client.deploy_model(modelingTaskID, bestModel)
    status = 'Failed'
    if str(depID) != '':
        status = 'Success'

    # Binding DeploymentID, Status and Best Model together in a dictionary to return as a output
    out_result = {DEPLOYMENT_ID: depID, STATUS: status, MODEL: bestModel}
    final_out = client.convert_json_to_dataframe(out_result)
    print('DeploymentID:' + str(depID))

    if platform == 'tabprep':
        return out_result

    prediction = []

    # Number of records in input
    max_length = len(data.index)
    max_data_length = len(data.index)

    # Adding survived to a list
    for i in range(0, max_data_length):
        prediction.append(data.iloc[i][label])

    print('returning result')
    return prediction

def rapidminer_score(go_url, gouser, gopassword, inputScoreData, label, depID):
    from rapidminer_go_python import rapidminergoclient as amw
    global client
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)

    print('Inside Score Method, DeploymentID '+str(depID))
    PREDICTION = 'prediction(' + label + ')'


    # passing the test data to deployed model to score
    scoreResult = client.score(inputScoreData, depID)

    # converting result json to dataframe
    result = client.convert_json_to_dataframe(scoreResult['data'])
    req = client.convert_json_to_dataframe(inputScoreData)

    # List to add the result data
    prediction = []

    # Number of records in input
    max_length = len(req.index)

    # Adding confidence and predictions to a list
    for i in range(0, max_length):
        prediction.append(result.iloc[i][PREDICTION])

    print('Scoring Completed Successfully')
    return prediction



def rapidminer_train_and_score(go_url, gouser, gopassword, input_train_data, input_score_data, label, platform):
    # List to add the result data
    prediction = []

    prediction.extend(rapidminerTrain(go_url, gouser, gopassword, input_train_data, label, platform))

    prediction.extend(rapidminerScore(go_url, gouser, gopassword, input_score_data, label, depID))
    print('returning result')
    return prediction



print('Deploying Quick Model')
tabclient.deploy('Rapidminer_Quick_Training',
              rapidminer_quick_training,
              'Quickly Trains a model for predictions', override=True)

print('Deploying Training and Score')
tabclient.deploy('RapidMiner_Train_And_Score',
              rapidminer_train_and_score,
              'Trains and Returns a dataset with predictions', override=True)


print('Deploying Score')
tabclient.deploy('RapidMiner_Score',
              rapidminer_score,
              'Returns a dataset with predictions', override=True)



print('Deploying Training Function ')
tabclient.deploy('RapidMiner_Train',
              rapidminer_train,
              'Trains a model for predictions', override=True)
