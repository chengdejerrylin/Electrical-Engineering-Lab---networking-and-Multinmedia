import os
import os.path
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre_model") )

import sklearn.datasets as datasets
import numpy as np
import pickle, gzip
import json
import packagedModel as pack
import csv
import numpy as np

def getAbsPath(path) :
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

def getCircles(size):
    X, y = datasets.make_circles(size, factor=.5, noise=.05)
    return X, [ [i] for i in y]

def getMnist():
    f = gzip.open(getAbsPath('mnist.pkl.gz'),'rb')
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
    with open(getAbsPath(path)) as f :
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

def getMnistModel() :
    import pretrain_mnist as mnist
    result = pack.classifyModel(mnist.layers)
    result.load(getAbsPath("pre_model/mnist.model"))
    return result

def getBigML(model_name, ratio = 1.0, path = "") :
    if not path : path = "../BigML/data/predict_result/" + model_name + "/" + model_name + "_results"
    path = getAbsPath(path)
    
    if model_name == "MNIST" or model_name == "MNIST_deepnet":
        input_number = 784
        output_category = 10
    elif model_name == "iris" or model_name == "iris_deepnet":
        input_number = 4
        output_category = 3


    input_list = []
    answer_list = []
    output_list = []
    predict_list = []
    with open(path) as readcsv :
        read_file = csv.reader(readcsv, delimiter=',')

        for i in range(output_category) :
            input_list.append([])
            answer_list.append([])
            output_list.append([])
            predict_list.append([])

        output_order = dict()
        title = True

        for row in read_file:
            if title :
                for i in range(output_category) :
                    output_order[row[input_number+2+i][:-12]] = i

                title = False
            else :
                order = output_order[row[input_number]]
                input_list[order].append( [float(data) for data in row[0:input_number] ] )
                answer_list[order].append(output_order[row[input_number]])
                output_list[order].append([float(data) for data in row[(int(input_number)+2) : (int(input_number) + int(output_category)+2)]])
                predict_list[order].append(output_order[row[input_number+1]])

    trainData, testData = [], []
    trainAns , testAns  = [], []
    trainProb, testProb = [], []
    trainPred, testPred = [], []

    for i in range(output_category) :
        n = int(len(input_list[i])*ratio + 0.5)
        choice = np.random.choice(len(input_list[i]), n , replace=False)
        mask = np.zeros(len(input_list[i]), dtype = bool)
        mask[choice] = True

        d, a, p, pr = np.array(input_list[i])[mask], np.array(answer_list[i])[mask], np.array(output_list[i])[mask], np.array(predict_list[i])[mask]
        d1, a1, p1, pr1 = np.array(input_list[i])[~mask], np.array(answer_list[i])[~mask], np.array(output_list[i])[~mask], np.array(predict_list[i])[~mask]

        for j in range(len(d)):
            trainData.append(d[j])
            trainAns.append(a[j])
            trainProb.append(p[j])
            trainPred.append(pr[j])

        for j in range(len(d1)):
            testData.append(d1[j])
            testAns.append(a1[j])
            testProb.append(p1[j])
            testPred.append(pr1[j])

    return trainData, trainAns, trainProb, trainPred, testData, testAns, testProb