import datasets
import packagedModel as pack

x_train, y_train = datasets.getTrainedIris()

layers = [("Linear", (5,5)), \
        ("ReLU", ()), \
        ("Linear", (5,5)), \
        ("ReLU", ()), \
        ("Linear", (5,3)), \
        ("Softmax", 1)]

model = pack.torchModel(layers, loss_func="MSELoss", optimArgs = {"lr" : 1e-3})
model.train(x_train, y_train, epoch=100, batch = 3)
model.save("iris.model")

for i in range(len(x_train)) :
	print()
	print("data", i+1)
	print("predict  :", model.predict(x_train[i])[0])
	print("actually :", y_train[i])
