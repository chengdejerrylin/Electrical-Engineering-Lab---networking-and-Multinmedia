import os
import os.path
import numpy as np
import datasets
import packagedModel as pack
from argparse import ArgumentParser

parser = ArgumentParser(description='A program that tries to steal the mnist classification model on BigML.')
parser.add_argument('-nt', '-nTrain', dest='nTrain'   , type=int  , default=200 , help='traring size.')
parser.add_argument('-nb', '-nBatch', dest='nBatch'   , type=int  , default=100 , help='traring batch size.')
parser.add_argument('-e' , '-epoch' , dest='epoch'    , type=int  , default=200 , help='epoch.')
parser.add_argument('-lr',            dest='lr'       , type=float, default=1e-4, help='Learning rate.')
parser.add_argument('-pPerEpoch',     dest='pPerEpoch', type=int  , default=10  , help='print per epoch.')
parser.add_argument('-noP',           dest='pDetail'  , action='store_false', help='do not print detail data.')

args = parser.parse_args()
nTrain = args.nTrain
epoch = args.epoch
nBatch = args.nBatch
lr = args.lr
pPerEpoch = args.pPerEpoch if args.pDetail else -1

x_train, y_train, y_value = datasets.getBigML("MNIST", nTrain)
x_test, y_test, _ = datasets.getBigML("MNIST", 2000)

layers = [("Linear", (784,100)), \
        ("ReLU", ()), \
        ("Linear", (100,100)), \
        ("ReLU", ()), \
        ("Linear", (100,10)), \
        ("Softmax", 1)]

#control model
if args.pDetail : print("\n======================== control model ========================")
control = pack.classifyModel(layers, optimArgs = {"lr" : lr})
control.train(x_train, y_train, epoch, nBatch, printPerEpoch=pPerEpoch, yType = "long", yTo2D = False, printData = args.pDetail, printAcc = args.pDetail)
control.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/origin_mnist_BigML.model"))

#copy model
if args.pDetail : print("\n======================== copy model ========================")
copy = pack.classifyModel(layers, loss_func="BCELoss", optimArgs = {"lr" : lr})
copy.train(x_train, y_value, epoch, nBatch, printPerEpoch=pPerEpoch, printData = args.pDetail, printAcc = args.pDetail)
copy.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/copy_mnist_BigML.model"))

print("\n======================== summary ========================")
if args.pDetail :
    print("control model")
    print(control)
    print("\ncopy model")
    print(copy)
print("\nSize of Training data  :", nTrain)
print("Size of Training batch :", nBatch)
print("Size of Testing  data  :", len(x_test))
print("Total Epoch :", epoch)
print("Learning Rate :", lr)
print("Training Accuracy of control model :", control.getAccuracy(x_train, y_train))
print("Training Accuracy of   copy model :", copy.getAccuracy(x_train, y_train))
print("Testing  Accuracy of control model :", control.getAccuracy(x_test, y_test))
print("Testing  Accuracy of   copy model :", copy.getAccuracy(x_test, y_test))
