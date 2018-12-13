import datasets
import packagedModel as pack

x_train, y_train = datasets.getIncome()

layers = [("Linear", (15,7)), \
        ("ReLU", ()), \
        ("Linear", (7,7)), \
        ("ReLU", ()), \
        ("Linear", (7,2)), \
        ("Softmax", 1)]

# layers = [("Linear", (4,2)), \
#         ("ReLU", ()), \
#         ("Linear", (2,3)), \
#         ("Softmax", 1)]

# layers = [("Linear", (5,3)), \
#         ("Softmax", 1)]

model = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 1e-4} )
model.train(x_train, y_train, epoch=1000, batch = 50, printPerEpoch=50)
model.save("income.model")
print(model.getAccuracy(x_train, y_train))

