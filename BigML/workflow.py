from bigml.model import Model
from bigml.api import BigML
import bigml.util as util

# from models import models, black_box_models
REQUEST_STORAGE = './data/request_result'

BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"
BB_USER_NAME = "HowardChao"
BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"

#### CREAT API
### All step that make request through api will be stored
api = BigML(storage=REQUEST_STORAGE)


#### CREAT MODEL
### api.ok() is to make sure each step is finished before running subsequent data.
### Create data source (from local)
source = api.create_source('./data/local_data/iris.csv')
api.ok(source)
dataset = api.create_dataset(source)
api.ok(dataset)
model = api.create_model(dataset)
api.ok(model)
### To set the target objective_field.
# model = api.create_model(dataset, {"objective_field": "species"})
### Model ecaluation
# evaluation = api.create_evaluation(model, test_dataset)
# api.ok(evaluation)

### You can call the model from 1. API 2. local model

# 1. This is for one input data
input_data = {"petal width": 1.75, "petal length": 2.45}
prediction = api.create_prediction(model, input_data)

print(prediction)
api.pprint(prediction)


# 2. This is for a bunch of input data 
test_source = api.create_source("./data/test_iris.csv")
api.ok(test_source)
test_dataset = api.create_dataset(test_source)
api.ok(test_dataset)
batch_prediction = api.create_batch_prediction(model, test_dataset, {"all_fields": True})
api.ok(batch_prediction)
api.download_batch_anomaly_score(batch_prediction, filename="REQUEST_STORAGE")