from bigml.model import Model
from bigml.api import BigML
import bigml.util as util
import os

from models import models, black_box_models

# from models import models, black_box_models
MODEL_STORAGE = './data/BigML_models'
DATASET_STORAGE = './data/datasets'

BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"
BB_USER_NAME = "HowardChao"
BB_KEY = "ce4f274a13375e6ffa0c4b321761dd67376cf435"


def main():
    pass

def model_creation(model_name, local_or_online):
    #### CREAT API
    ### All step that make request through api will be stored
    model_storage = os.path.join(MODEL_STORAGE, model_name)
    if not os.path.exists(model_storage):
        os.makedirs(model_storage)
    api = BigML(storage=model_storage)
    if local_or_online == "L":
        #### CREAT MODEL
        ### api.ok() is to make sure each step is finished before running subsequent data.
        ### Create data source (from local)
        print("Creating model from Local data .... ")
        dataset_storage = os.path.join(DATASET_STORAGE, model_name, model_name+"_train.csv")
        print("Reading training data .... ")
        source = api.create_source(dataset_storage)
        api.ok(source)
        dataset = api.create_dataset(source)
        api.ok(dataset)
        print("Model creating .... ")
        if which_model == "D":
            model = api.create_model(dataset)
        elif which_model == "E":
            model = api.create_ensemble(dataset)
        elif which_model == "DN":
            model = api.create_deepnet(dataset)
        elif which_model == "A":
            model = api.create_association(dataset)
        else:
            print("Your input model is invalid, byebye!!!")
            sys.exit()
        api.ok(model)
        print("Model is created ! DONE !!")
        print(">> model name : ", model_name, "  model id : ", model["resource"])
    elif local_or_online == "O":
        #### DOWNLOAD MODEL
        print("Download model from API .... ")
        print(">> model name : ", model_name, "  model id : ", models[model_name])
        api = BigML(storage=model_storage)
        api.export('model/{}'.format(models[model_name]))

if __name__ == "__main__":
    main()
    model_name = input('Enter a model name: ')
    local_or_online = input('Local or online (L/O)?')
    which_model = input('Which model (Decison_tree / Ensemble / Logistic_regression / Association_model)?')
    model_creation(model_name, local_or_online)