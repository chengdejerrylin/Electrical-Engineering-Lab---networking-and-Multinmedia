import os
import os.path
import numpy as np
import datasets
import packagedModel as pack

nTrain = 50
epoch = 100
nBatch = 25
pPerEpoch = 10

x_train, y_train, y_value = datasets.getBigML("MNIST", nTrain)
x_test, y_test, _ = datasets.getBigML("MNIST", 2000)

layers = [("Linear", (784,100)), \
        ("ReLU", ()), \
        ("Linear", (100,100)), \
        ("ReLU", ()), \
        ("Linear", (100,10)), \
        ("Softmax", 1)]

#Origin model
print("\n======================== origin model ========================")
origin = pack.classifyModel(layers, optimArgs = {"lr" : 5e-5})
origin.train(x_train, y_train, epoch, nBatch, printPerEpoch=pPerEpoch, yType = "long", yTo2D = False)
origin.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/origin_mnist_BigML.model"))

#copy model
print("\n======================== copy model ========================")
copy = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 5e-5})
copy.train(x_train, y_value, epoch, nBatch, printPerEpoch=pPerEpoch)
copy.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/copy_mnist_BigML.model"))


print("\n======================== summary ========================")
print("origin model")
print(origin)
print("\ncopy model")
print(copy)
print("\nSize of Training data  :", nTrain)
print("Size of Training batch :", nBatch)
print("Size of Testing  data  :", len(x_test))
print("Total Epoch :", epoch)
print("Training Accuracy of origin model :", origin.getAccuracy(x_train, y_train))
print("Training Accuracy of   copy model :", copy.getAccuracy(x_train, y_train))
print("Testing  Accuracy of origin model :", origin.getAccuracy(x_test, y_test))
print("Testing  Accuracy of   copy model :", copy.getAccuracy(x_test, y_test))
