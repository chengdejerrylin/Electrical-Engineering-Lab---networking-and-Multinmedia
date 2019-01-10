MNIST = [("Linear", (784,100)), \
        ("ReLU", ()), \
        ("Linear", (100,100)), \
        ("ReLU", ()), \
        ("Linear", (100,10)), \
        ("Softmax", 1)]

def getLayer(name) :
	if name == "MNIST" or name == "MNIST_deepnet" : return MNIST