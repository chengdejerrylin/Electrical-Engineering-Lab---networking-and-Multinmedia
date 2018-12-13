import datasets
import packagedModel as pack

x_train, y_train = datasets.getTrainedIris()

# layers = [("Linear", (5,5)), \
#         ("ReLU", ()), \
#         ("Linear", (5,5)), \
#         ("ReLU", ()), \
#         ("Linear", (5,3)), \
#         ("Softmax", 1)]

layers = [("Linear", (5,2)), \
        ("ReLU", ()), \
        ("Linear", (2,3)), \
        ("Softmax", 1)]

# layers = [("Linear", (5,3)), \
#         ("Softmax", 1)]

model = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 1e-3} )
model.train(x_train, y_train, epoch=2000, batch = len(x_train)//2, printPerEpoch=100)
model.save("iris.model")
print(model.getAccuracy(x_train, y_train))

