import sklearn.datasets as datasets
import pickle, gzip

def getCircles(size):
    X, y = datasets.make_circles(size, factor=.5, noise=.05)
    return X, [ [i] for i in y]

def getMnist():
	f = gzip.open('mnist.pkl.gz','rb')
	train_set, valid_set, test_set = pickle.load(f,encoding='unicode-escape')
	f.close()
	return train_set[0], train_set[1], test_set[0], test_set[1]

def getIris():
	return datasets.load_iris(True)