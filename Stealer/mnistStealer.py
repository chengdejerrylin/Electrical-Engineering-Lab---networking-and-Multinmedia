import os
import os.path
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre_model") )

import numpy as np
import datasets
import packagedModel as pack

nTrain = 200
epoch = 100
nBatch = 25
pPerEpoch = 10

print("\n======================== pretrained model========================")
pre_model = datasets.getMnistModel()
print(pre_model.model)

x_train, y_train , x_test, y_test = datasets.getMnist()
mask = np.random.choice(x_train.shape[0], nTrain , replace=False)
x_train = x_train[mask]
y_train = y_train[mask]
y_value = pre_model.predict(x_train)

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
origin.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/origin_mnist.model"))

#copy model
print("\n======================== copy model ========================")
copy = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 5e-5})
copy.train(x_train, y_value, epoch, nBatch, printPerEpoch=pPerEpoch)
copy.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/copy_mnist.model"))


print("\n======================== summary ========================")
print("Training Accuracy of origin model :", origin.getAccuracy(x_train, y_train))
print("Training Accuracy of   copy model :", copy.getAccuracy(x_train, y_train))
print("Testing  Accuracy of origin model :", origin.getAccuracy(x_test, y_test))
print("Testing  Accuracy of   copy model :", copy.getAccuracy(x_test, y_test))