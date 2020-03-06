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
AUTODEPLOY = False
client = ''

tabclient = Client('http://localhost:9004/')
#new update

def rapidminer_quick_autommodel(go_url, gouser, gopassword, input_data, label, platform):
    print('URL:'+go_url)
    from rapidminer_go_python import rapidminergoclient as amw
    LABEL_ATTRIBUTE = label
    # To refresh the changes made in AutoModelWeb
    #sys.path.append(os.path.dirname('C:/CodeBase/rapidminer-go-python/mar06/rapidminer_go_python'))
    #import rapidminergoclient as amw


    # To refresh the changes made in AutoModelWeb
    importlib.reload(amw)

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

def rapidminerTrain(go_url, gouser, gopassword, input_data, label, platform):

    from rapidminer_go_python import rapidminergoclient as amw
    LABEL_ATTRIBUTE = label

   # To refresh the changes made in AutoModelWeb
    #sys.path.append(os.path.dirname('C:/CodeBase/rapidminer-go-python/mar06/rapidminer_go_python'))
    #import rapidminergoclient as amw
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)
    # To get the AMW instance
    #connection(go_url, gouser, gopassword)

    data = client.convert_json_to_dataframe(input_data)

    # dataframe to json
    dataId = client.add_dataFrame(data)[DATA_ID]


    modelingTaskID = client.create_modeling_task(dataId)[DATA_ID]


    # setting label
    client.set_label(modelingTaskID, LABEL_ATTRIBUTE)
    client.set_class_interest(modelingTaskID, 'Yes', 'No')
    jsonVal = client.set_cost_matrix(modelingTaskID, [[1, -1], [-1, 1]])
    print('TaskID:' + modelingTaskID)

    # Initiating model training
    client.start_execution(modelingTaskID)
    print('ExecutingModel...')
    # Loops till all the task are completed
    #client.wait_till_execution_completion(modelingTaskID)

    #print('Model Execution Completed')

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

def rapidminerScore(go_url, gouser, gopassword, inputScoreData, label, depID):
    from rapidminer_go_python import rapidminergoclient as amw
    print("URL is " + go_url)
    global client
    #sys.path.append(os.path.dirname('C:/CodeBase/rapidminer-go-python/mar06/rapidminer_go_python'))
    #import rapidminergoclient as amw
    client = amw.RapidMinerGoClient(go_url, gouser, gopassword)

    print('Inside Score Method, DeploymentID'+str(depID))
    PREDICTION = 'prediction(' + label + ')'


    # passing the test data to deployed model to score
    scoreResult = client.score(inputScoreData, depID)

    # converting result json to dataframe
    result = client.convert_json_to_dataframe(scoreResult['data'])
    req = client.convert_json_to_dataframe(inputScoreData)
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



def rapidminerTrainAndScore(go_url, gouser, gopassword, input_train_data, input_score_data, label, platform):
    # List to add the result data
    prediction = []

    prediction.extend(rapidminerTrain(go_url, gouser, gopassword, input_train_data, label, platform))

    prediction.extend(rapidminerScore(go_url, gouser, gopassword, input_score_data, label, depID))
    print('returning result')
    return prediction



print('Deploying Quick Model')
tabclient.deploy('Rapidminer_Quick_Automodel',
              rapidminer_quick_autommodel,
              'Quickly Trains a model for predictions', override=True)

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
              'Trains a model for predictions', override=True)
