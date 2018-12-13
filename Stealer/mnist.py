import packagedModel as pack
import datasets


x_train, y_train, x_test, y_test = datasets.getMnist()

layers = [("Linear", (784,100)), \
        ("ReLU", ()), \
        ("Linear", (100,100)), \
        ("ReLU", ()), \
        ("Linear", (100,10)), \
        ("Softmax", 1)]


model = pack.classifyModel(layers, loss_func="CrossEntropyLoss", optimArgs = {"lr" : 5e-5})
model.train(x_train, y_train, 10, 100)
model.save("minst.model")
print(model.getAccuracy(x_train, y_train))