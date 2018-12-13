import datasets
import packagedModel as pack

x_train, y_train = datasets.getTrainedIris()

layers = [("Linear", (5,5)), \
        ("ReLU", ()), \
        ("Linear", (5,5)), \
        ("ReLU", ()), \
        ("Linear", (5,3)), \
        ("Softmax", 1)]

# layers = [("Linear", (5,3)), \
#         ("ReLU", ()), \
#         ("Linear", (3,3)), \
#         ("Softmax", 1)]


model = pack.classifyModel(layers, loss_func="MSELoss", optimArgs = {"lr" : 1e-6} )
model.train(x_train, y_train, epoch=500, batch = len(x_train)//4, printPerEpoch = 50)
model.save("iris.model")
print(model.getAccuracy(x_train, y_train))
