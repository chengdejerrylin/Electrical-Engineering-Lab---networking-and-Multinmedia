import os
import os.path
import datasets
import packagedModel as pack
import numpy as np

nTrain = 13
epoch = 50
nBatch = 13
pPerEpoch = 1
trainTestRatio = .5

x, y, y_prob = datasets.getBigML("iris_deepnet")
x, y, y_prob = np.array(x), np.array(y), np.array(y_prob)

boundary = int(len(x)*trainTestRatio)
mask = np.random.choice(boundary, nTrain , replace=False)
x_train, y_train, y_value = x[mask], y[mask], y_prob[mask]
x_test, y_test = x[boundary:], y[boundary:]

layers = [("Linear", (4,4)), \
        ("ReLU", ()), \
        ("Linear", (4,3)), \
        ("Softmax", 1)]

#Origin model
print("\n======================== origin model ========================")
origin = pack.classifyModel(layers, optimArgs = {"lr" : 1e-4})
origin.train(x_train, y_train, epoch, nBatch, printPerEpoch=pPerEpoch, yType = "long", yTo2D = False)
origin.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/origin_mnist_BigML.model"))

#copy model
print("\n======================== copy model ========================")
copy = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 1e-4})
copy.train(x_train, y_value, epoch, nBatch, printPerEpoch=pPerEpoch)
copy.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/copy_mnist_BigML.model"))


print("\n======================== summary ========================")
print("Training Accuracy of origin model :", origin.getAccuracy(x_train, y_train))
print("Training Accuracy of   copy model :", copy.getAccuracy(x_train, y_train))
print("Testing  Accuracy of origin model :", origin.getAccuracy(x_test, y_test))
print("Testing  Accuracy of   copy model :", copy.getAccuracy(x_test, y_test))

