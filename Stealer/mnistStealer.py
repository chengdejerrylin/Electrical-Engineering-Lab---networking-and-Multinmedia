import os
import os.path
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre_model") )

import numpy as np
import datasets
import packagedModel as pack

nTrain = 10000
epoch = 100
nBatch = nTrain//5

pre_model = datasets.getMnistModel()

x_train, _ , x_test, y_test = datasets.getMnist()
x_train = x_train[np.random.choice(x_train.shape[0], nTrain , replace=False)]
y_train = pre_model.predict(x_train)

layers = [("Linear", (784,100)), \
        ("ReLU", ()), \
        ("Linear", (100,100)), \
        ("ReLU", ()), \
        ("Linear", (100,10)), \
        ("Softmax", 1)]

model = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 5e-5})
model.train(x_train, y_train, epoch, nBatch, printPerEpoch=10)
model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/mnist.model"))

print(model.getAccuracy(x_test, y_test))