import torch as t 
import numpy as np

use_cuda = t.cuda.is_available()

class torchModel(object):

    def __init__(self, model = [], optim = "Adam", loss_func = "binary_cross_entropy", optimArgs = dict()):
        
        if model :
            self.model = self._getModel(model)
            self.optim = getattr(t.optim, optim)( self.model.parameters(), **optimArgs)
            if use_cuda : self.model = self.model.cuda()
        
        try:
            self.loss_func = getattr(t.nn, loss_func)()
            if use_cuda : self.loss = self.loss.cuda()
        except AttributeError:
            self.loss_func = getattr(t.nn.functional, loss_func)
        except AttributeError:
            self.loss_func = loss_func



        self.optimArgs = optimArgs

    def predict(self, x, xType = "") :
        return self.model(self._inputTransform(x, xType)).detach().numpy()

    def train(self, x_train, y_train, epoch = 1000 , batch = 500, xType = "", yType = "", printPerEpoch = 1) :
        x, y = self._inputTransform(x_train, xType), self._inputTransform(y_train, yType)
        nTrain = x.size()[0]

        #test type
        mask = [0, 1]
        y_test = self.model(x[mask])
        try:
            loss = self.loss_func(y_test, y[mask])
        except Exception as e:
            y = self._inputTransform(y_train, yType, False)

        print("training model...")
        print("Epoch:", epoch, ",Training_data_size:", nTrain, ",Batch_size:", batch)
        print("optimizer:", self.optim)
        print("loss_function:", self.loss_func, end="\n\n")

        for e in range(epoch) :
            for b in range(nTrain // batch) :
                #batch and predict
                mask = np.random.choice(nTrain, batch , replace=False)
                x_batch, y_batch = x[mask], y[mask]
                y_pred = self.model(x_batch)

                #loss and feedback
                loss = self.loss_func(y_pred , y_batch)

                self.model.zero_grad()
                loss.backward()
                self.optim.step()

            if (e+1) % printPerEpoch == 0 :
                print("Epoch:", e+1, ",loss:", float(loss.detach()), ",Accuracy:", self.getAccuracy(x_batch, y_batch))

    def getNWeight(self) :
        result = 0
        for par in self.model.parameters() :
            
            s = 1
            for dimention in par.data.size() : s *= dimention
            result += s

        return result

    def save(self, path) :
        t.save(self.model.state_dict(), path)

    def load(self, path) :
        self.model.load_state_dict(t.load(path))
        if use_cuda : self.model = self.model.cuda()

        self.optim = getattr(t.optim, optim)( self.model.parameters(), **self.optimArgs)

    def _getModel(self, layers):
        try:
            
            result = []

            for l in layers : 
                try:
                    result.append( getattr(t.nn, l[0])(*l[1]) )
                except Exception as e:
                    try:
                        result.append( getattr(t.nn, l[0])( l[1]) )
                    except Exception as e:
                        result.append( getattr(t.nn, l[0])( **l[1]) )
                

            return t.nn.Sequential(*result)

        except Exception as e:
            return layers

    def _inputTransform(self, x, xType = "", to2D = True) :

        result = x
        if type(result) != type(t.tensor([0])) : 
            
            if type(result) == type([]) : result = np.array(result) #list to numpy
            if to2D : #1d to 2d
                #if len(np.shape(result)) == 1 : result = np.array([result]) 
                if len(np.shape(result)) == 1 : result = np.array([ [i] for i in result])

            #numpy to tensor
            temp = np.array(result)
            result = t.from_numpy(result)
            if use_cuda : result.cuda()

            # float64 to float
            try:
                if type(temp[0][0]) == type(np.array([0.1])[0]) : result = result.float() 
            except Exception as e:
                if type(temp[0]) == type(np.array([0.1])[0]) : result = result.float()
            

            #change type
            if xType : result = getattr(result, xType)()

        return result

    def getAccuracy(self, x, y) : return -1
    def getAccuracyFromProb(self, x, y) : return -1

class classifyModel(torchModel):
    """docstring for ClassifyModel"""
    def __init__(self, layers = [], optim = "Adam", loss_func = "CrossEntropyLoss", optimArgs = dict()):
        super(classifyModel, self).__init__(layers, optim, loss_func, optimArgs)

    def getAccuracy(self, x, y) :
        
        x, y = self._inputTransform(x), self._inputTransform(y)
        y_pred = t.max(self.model(x),1)[1]

        try: # y = index of answer
            return (y_pred==y).sum().item()/y.shape[0]

        except Exception as e: # y = prob of classes
            y = t.max(y,1)[1]
            return (y_pred==y).sum().item()/y.shape[0]
        
