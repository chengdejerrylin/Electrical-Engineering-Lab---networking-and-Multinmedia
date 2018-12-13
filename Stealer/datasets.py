import sklearn.datasets as datasets
import numpy as np
import pickle, gzip
import json

def getCircles(size):
    X, y = datasets.make_circles(size, factor=.5, noise=.05)
    return X, [ [i] for i in y]

def getMnist():
    f = gzip.open('mnist.pkl.gz','rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='unicode-escape')
    f.close()
    return train_set[0], train_set[1], test_set[0], test_set[1]

def getIris():
    return datasets.load_iris(True)

def getTrainedIris() :
    return getDataFromProbabilityTxt("../BigML/data/predict_result/iris/API_result/probabilities.txt")

def getIncome() :
    return getDataFromProbabilityTxt("../BigML/data/predict_result/income/API_result/probabilities.txt")
    
def getDataFromProbabilityTxt(path) :
    with open(path) as f :
        data = f.read()

    data = data.replace("\'", "\"").replace("None", "0")
    data = json.loads(data)

    x, y = [], []

    table = dict()

    for oneData in data :
        x_temp, y_temp = [], []

        for key, value in oneData.items() :

            if key == "probability" : 
                for name, prob in value.items() : y_temp.append(float(prob) )

            else :
                try:
                    x_temp.append(float(value) )
                except ValueError:
                    
                    if key not in table : table[key] = dict() 
                    if value not in table[key] : table[key][value] = float(len(table[key]))

                    x_temp.append(table[key][value])

        x.append(x_temp)
        y.append(y_temp)

    return np.array(x), np.array(y)
