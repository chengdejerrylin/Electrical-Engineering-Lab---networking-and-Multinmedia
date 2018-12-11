from bigml.model import Model
from bigml.api import BigML
import bigml.util as util
import os

# from models import models, black_box_models
MODEL_STORAGE = './data/BigML_models'
DATASET_STORAGE = './data/datasets'

BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"
BB_USER_NAME = "HowardChao"
BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"


def main():
    pass

def model_creation(model_name):
    #### CREAT API
    ### All step that make request through api will be stored
    model_storage = os.path.join(MODEL_STORAGE, model_name)
    if not os.path.exists(model_storage):
        os.makedirs(model_storage)
    api = BigML(storage=model_storage)

    #### CREAT MODEL
    ### api.ok() is to make sure each step is finished before running subsequent data.
    ### Create data source (from local)
    dataset_storage = os.path.join(DATASET_STORAGE, model_name, model_name+"_train.csv")
    source = api.create_source(dataset_storage)
    api.ok(source)
    dataset = api.create_dataset(source)
    api.ok(dataset)
    model = api.create_model(dataset)
    api.ok(model)

if __name__ == "__main__":
    main()
    model_name = input('Enter a model name: ')
    model_creation(model_name)