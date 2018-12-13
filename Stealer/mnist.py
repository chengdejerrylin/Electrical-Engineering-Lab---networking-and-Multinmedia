import packagedModel as pack
import datasets
import numpy as np


x_train, y_train, x_test, y_test = datasets.getMnist()

layers = [("Linear", (784,100)), \
        ("ReLU", ()), \
        ("Linear", (100,100)), \
        ("ReLU", ()), \
        ("Linear", (100,10)), \
        ("Softmax", 1)]

epoch = 10
batch = 100


model = pack.classifyModel(layers, loss_func="CrossEntropyLoss", optimArgs = {"lr" : 5e-5})
model.train(x_train, y_train, epoch, batch)
model.save("minst.model")

mask = np.random.choice(len(x_train), batch, replace = False)
print(model.getAccuracy(x_train, y_train))