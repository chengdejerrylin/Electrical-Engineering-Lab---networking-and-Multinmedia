import datasets
import packagedModel as pack

x_train, y_train = datasets.getTrainedIris()
x_test, y_test = datasets.getIris()

# layers = [("Linear", (5,5)), \
#         ("ReLU", ()), \
#         ("Linear", (5,5)), \
#         ("ReLU", ()), \
#         ("Linear", (5,3)), \
#         ("Softmax", 1)]

layers = [("Linear", (4,2)), \
        ("ReLU", ()), \
        ("Linear", (2,3)), \
        ("Softmax", 1)]

# layers = [("Linear", (5,3)), \
#         ("Softmax", 1)]

model = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : 1e-3} )
model.train(x_train, y_train, epoch=1000, batch = len(x_train)//2, printPerEpoch=50)
model.save("iris.model")

for i in x_test :
    print(model.predict([i]))

