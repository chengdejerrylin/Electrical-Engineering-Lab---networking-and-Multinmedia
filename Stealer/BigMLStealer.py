import os
import os.path
import numpy as np
import datasets
import packagedModel as pack
import layers
from argparse import ArgumentParser

def getAbsPath(path) :
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

parser = ArgumentParser(description='A program that tries to steal the classification model on BigML.')
parser.add_argument('data', help='select the database.')
parser.add_argument('-r', '-ratio',   dest='ratio'    , type=float, default=0.1 , help='traring testing ratio.')
parser.add_argument('-nb', '-nBatch', dest='nBatch'   , type=int  , default=100 , help='traring batch size.')
parser.add_argument('-e' , '-epoch' , dest='epoch'    , type=int  , default=200 , help='epoch.')
parser.add_argument('-lr',            dest='lr'       , type=float, default=1e-4, help='Learning rate.')
parser.add_argument('-pPerEpoch',     dest='pPerEpoch', type=int  , default=10  , help='print per epoch.')
parser.add_argument('-noP',           dest='pDetail'  , action='store_false', help='do not print detail data.')
parser.add_argument('-loss',          dest='loss'     , default="BCELoss", help='do not print detail data.')

args = parser.parse_args()
ratio = args.ratio
epoch = args.epoch
nBatch = args.nBatch
lr = args.lr
pPerEpoch = args.pPerEpoch if args.pDetail else -1

x_train, y_train, y_value, x_test, y_test, _ = datasets.getBigML(args.data, ratio)
layer = layers.getLayer(args.data)

#control model
if args.pDetail : print("\n======================== control model ========================")
control = pack.classifyModel(layer, optimArgs = {"lr" : lr})
control.train(x_train, y_train, epoch, nBatch, printPerEpoch=pPerEpoch, yType = "long", yTo2D = False, printData = args.pDetail, printAcc = args.pDetail)
control.save(getAbsPath("model/BigML_" + args.data + "_control.model"))

#copy model
if args.pDetail : print("\n======================== copy model ========================")
copy = pack.classifyModel(layer, loss_func=args.loss, optimArgs = {"lr" : lr})
copy.train(x_train, y_value, epoch, nBatch, printPerEpoch=pPerEpoch, printData = args.pDetail, printAcc = args.pDetail)
copy.save(getAbsPath("model/BigML_" + args.data + "_copy.model"))

if args.pDetail :
    print("\n======================== summary ========================")
    print("control model")
    print(control)
    print("\ncopy model")
    print(copy)
    print("\nSize of Training data  :", len(x_train))
    print("Size of Training batch :", nBatch)
    print("Size of Testing  data  :", len(x_test))
    print("Total Epoch :", epoch)
    print("Learning Rate :", lr)
    print("loss function :", args.loss)
    print("Training Accuracy of control model :", control.getAccuracy(x_train, y_train))
    print("Training Accuracy of   copy model :", copy.getAccuracy(x_train, y_train))
    print("Testing  Accuracy of control model :", control.getAccuracy(x_test, y_test))
    print("Testing  Accuracy of   copy model :", copy.getAccuracy(x_test, y_test))
else :
    csv = str(len(x_train))
    csv += ',' + str(len(x_test))
    csv += ',' + args.loss
    csv += ',' + str(nBatch)
    csv += ',' + str(lr)
    csv += ',' + str(epoch)
    csv += ',' + str(control.getAccuracy(x_train, y_train))
    csv += ',' + str(copy.getAccuracy(x_train, y_train))
    csv += ',' + str(control.getAccuracy(x_test, y_test))
    csv += ',' + str(copy.getAccuracy(x_test, y_test))
    print(csv)
    
