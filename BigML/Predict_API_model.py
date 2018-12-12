from bigml.model import Model
from bigml.api import BigML
import bigml.util as util
import glob
import os
import csv 

from models import models, black_box_models


# from models import models, black_box_models
MODEL_STORAGE = './data/BigML_models'
DATASET_STORAGE = './data/datasets'
PREDICT_STORAGE = './data/predict_result'

BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"
BB_USER_NAME = "HowardChao"
BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"

def main():
    pass

def test_online_model(model_name):
    
    # Create local_model object
    print("Creating model from API .... ")

    predict_storage = os.path.join(PREDICT_STORAGE, model_name)
    if not os.path.exists(predict_storage):
        print("Creating predict directory .... ")
        os.makedirs(predict_storage)
    API_predict_storage = os.path.join(predict_storage, "API_result")
    if not os.path.exists(API_predict_storage):
        print("Creating predict directory .... ")
        os.makedirs(API_predict_storage)
    api = BigML(storage=API_predict_storage)
    print("Reading testing data .... ")
    test_source = api.create_source(os.path.join(DATASET_STORAGE, model_name, model_name+"_test.csv"))
    api.ok(test_source)
    test_dataset = api.create_dataset(test_source)
    api.ok(test_dataset)
    print("Start predicting .... ")
    print("    Opening testing data")
    training_data_path = os.path.join(DATASET_STORAGE, model_name, model_name) + "_test.csv"
    with open(training_data_path) as test_handler:
        reader = csv.DictReader(test_handler)
        counter = 1
        for input_data in reader:
            print("=================================")
            print("=====  Prediction ", counter, "  ========")
            print("=================================")
            print("Input testing data : ", input_data)
            predict_result = api.create_prediction('model/{}'.format(models[model_name]), input_data)
            print(">> Prediction : ", predict_result, "\n")
            api.pprint(predict_result)
            counter = counter + 1
    # batch_prediction = api.create_batch_prediction('model/{}'.format(models[model_name]), test_dataset, {"all_fields": True})
    # api.ok(batch_prediction)
    # # print(batch_prediction)
    # api.pprint(batch_prediction)

    # predict_result = local_model.predict({"petal length": 2.45, "sepal length": 2})
    # Write read
    # print(predict_result)
    # batch_prediction = api.create_batch_prediction(local_model, test_dataset, {"all_fields": True})

    # api.ok(batch_prediction)
    # api.download_batch_anomaly_score(batch_prediction, filename=os.path.join(predict_storage, model_name + "results"))


if __name__ == "__main__":
    main()
    model_name = input('Enter a model name: ')
    test_online_model(model_name)