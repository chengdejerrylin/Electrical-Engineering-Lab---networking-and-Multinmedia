from bigml.model import Model
from bigml.api import BigML
import bigml.util as util
import glob
import os
import csv

# from models import models, black_box_models
MODEL_STORAGE = './data/BigML_models'
DATASET_STORAGE = './data/datasets'
PREDICT_STORAGE = './data/predict_result'

BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"
BB_USER_NAME = "HowardChao"
BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"

def main():
    pass

def test_local_model(model_name):
    
    # Create local_model object
    print("Creating local model from file .... ")
    model_file = glob.glob(os.path.join(MODEL_STORAGE, model_name, "model_*"))
    local_model = Model(model_file[0])
    
    predict_storage = os.path.join(PREDICT_STORAGE, model_name)
    if not os.path.exists(predict_storage):
        print("Creating predict directory .... ")
        os.makedirs(predict_storage)
    predict_storage_local = os.path.join(predict_storage, "local_model_result")
    if not os.path.exists(predict_storage_local):
        print("Creating predict directory .... ")
        os.makedirs(predict_storage_local)
    print("Start predicting .... ")
    print("    Opening testing data")
    training_data_path = os.path.join(DATASET_STORAGE, model_name, model_name) + "_test.csv"
    with open(training_data_path, 'r') as test_handler, open(os.path.join(predict_storage_local, "PREDICT.txt"), 'w') as fh:
        reader = csv.DictReader(test_handler)
        counter = 1
        tmp = ""
        for input_data in reader:
            tmp = tmp + "=================================\n"
            print("=================================")
            tmp = tmp + "=====  Prediction " + str(counter) + "  ========\n"
            print("=====  Prediction ", counter, "  ========")
            tmp = tmp + "=================================\n"
            print("=================================")
            tmp = tmp + "Input testing data : " + str(input_data) + "\n"
            print("Input testing data : ", input_data)
            predict_result = local_model.predict(input_data)
            tmp = tmp + ">> Prediction : " + str(predict_result) + "\n\n"
            print(">> Prediction : ", predict_result, "\n")
            fh.write(tmp)
            counter = counter + 1
        ### Write file !!!
    # batch_prediction = api.create_batch_prediction(local_model, test_dataset, {"all_fields": True})

    # api.ok(batch_prediction)
    # api.download_batch_anomaly_score(batch_prediction, filename=os.path.join(predict_storage, model_name + "results"))


if __name__ == "__main__":
    main()
    model_name = input('Enter a model name: ')
    test_local_model(model_name)