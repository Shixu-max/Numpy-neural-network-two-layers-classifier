# two layers classifier
#Author: Xu Shi 22110980014
import numpy as np
from tqdm import tqdm # progress bar

# activation function and its derivative
class Activation():
    def __init__(self):
        pass

    @classmethod
    def activation_func(self, name = 'relu'):
        if name == 'relu':
            return lambda x: np.where(x > 0 , x , 0) # relu = max(0,x)
        elif name == 'sigmoid':
            return lambda x: 1. / (1 + np.exp(-np.clip(x, -30, 30)))# sig = 1/(1+e^(-x))

    @classmethod
    # derivative of sigmoid
    def derivative_sigmoid(self, x):
        t = 1. / (1 + np.exp(-np.clip(x, -30, 30))) # sig = 1/(1+e^(-x))
        return t * (1. - t) # sig' = sig(1-sig)

    @classmethod
    # derivative of relu and sigmoid
    def derivative(self, name = 'relu'):
        if name == 'relu':
            return lambda x: np.where(x > 0 , 1 , 0) # relu' = 0 (x<0);1 (x>0)
        elif name == 'sigmoid':
            return Activation.derivative_sigmoid

# two layer classifier
class Classifier():
    # initialization
    def __init__(self, 
            hidden_size = [784,100,10], acts = ['relu','sigmoid'], regws = None,regbs = None, lr = 1e-2):
           # hidden layers size         activation function       L2-regularation of weights, biases # learning rate

        self.hidden_size = hidden_size
        self.weights = []
        self.biases = []
        self.acts = acts
    
        # initialization
        for w, h in zip(hidden_size[:-1], hidden_size[1:]):
            self.weights.append( np.random.randn(w, h) * .1 )
            self.biases .append( np.random.randn(1, h) * .1 )
            self.weights[-1] = self.weights[-1].astype('float32')
            self.biases[-1]  = self.biases[-1] .astype('float32')
        
        self.z = [None for _ in range(len(self.weights)+1)]
        self.grads = [None for _ in range(len(self.weights)+1)]
        self.regws = [0] * len(self.weights) if regws is None else regws
        self.regbs = [0] * len(self.weights) if regbs is None else regbs
        self.lr = lr
        
    # forward pass
    def forward(self, x):
        self.z[0] = x
        for i in range(len(self.acts)):
            self.z[i+1] = self.z[i] @ self.weights[i] + self.biases[i]
            self.z[i+1] = Activation.activation_func(self.acts[i])(self.z[i+1])
        return self.z[-1]
    
    def __call__(self, x):
        return self.forward(x)

    def backprop(self, dx):
        # backpropagation
        for i in range(len(self.acts)-1, -1, -1):
            # derivative of activator 
            dx = Activation.derivative(self.acts[i])(self.z[i+1]) * dx
            self.grads[i] = self.z[i].T @ dx
            dx = dx @ self.weights[i].T

    # SGD
    def update(self, lr):
        for i in range(len(self.weights)):
            if self.grads[i] is not None:
                if self.regws[i] != 0:
                    self.weights[i] -= self.weights[i] * self.regws[i]
                if self.regbs[i] != 0:
                    self.biases [i] -= self.biases [i] * self.regbs[i]
                self.weights[i] -= self.grads[i] * lr

    def compute_loss(self, pred, fact, batch_size, loss_func):
        # compute loss
        loss = 0
        if loss_func == 0:   # MSE
            loss = .5 * np.sum(np.square(pred - fact)) / batch_size
        elif loss_func == 1: # BCE
            loss = -1. / batch_size * np.sum(
                fact*np.log(pred) + (1 - fact)*np.log((1+1e-7) - pred))
            # Here we add 1e-7 to prevent log(0)
            
        # L2-regularization loss
        for i in range(len(self.weights)):
            if self.regws[i] != 0:
                loss += .5 * self.regws[i] * np.sum(np.square(self.weights[i]))
            if self.regbs[i] != 0:
                loss += .5 * self.regbs[i] * np.sum(np.square(self.biases [i]))

        return loss

    # train data x,y
    def trmodel(self, x, y, epochs = 1, batch_size = 40, loss_func = 'MSE', valid_x = None, 
                valid_y = None, valid_freq = 1, pre_result = None, accs = None, lr = None): 
        # epochs: size of epochs
        # batch_size: size od batch on SGD
        # valid_freq: validate on the validation data every epoch
        # pre_result: the training will append the result in the dictionary
        # lr: learn rate

        # default
        if lr is not None: 
            self.lr = lr
        loss_func = {'MSE': 0, 'BCE': 1}[loss_func]
        
        # inherit the results
        if pre_result is not None:
            losses = pre_result['loss']
            accs = pre_result['acc']
            losses_valid = pre_result['loss_valid']
        else:
            losses = []
            accs = []
            losses_valid = []

        n = x.shape[0] 
        for epoch in range(1, 1 + epochs):
            # random shuffle indices
            shuffle = np.arange(n)
            np.random.shuffle(shuffle)
            batch_e = 0
            for _ in tqdm(range(n // batch_size)):
                batch_s = batch_e 
                batch_e = batch_s + batch_size
                batch_x = x[shuffle[batch_s: batch_e]] 
                batch_y = y[shuffle[batch_s: batch_e]]

                # forward
                predict_y = self.forward(batch_x)

                # loss
                losses.append( self.compute_loss(predict_y, batch_y, batch_size, loss_func) )

                # backpropagation
                if loss_func == 0:   # MSE
                    self.backprop(predict_y - batch_y)
                elif loss_func == 1: # BCE
                    self.backprop( 
                        np.true_divide(predict_y - batch_y, predict_y * ((1+1e-5) - predict_y)))

                # update
                self.update(self.lr)

            if valid_x is not None and epoch % valid_freq == 0:
                # validate
                valid_result = self.predict(valid_x, valid_y, batch_size, 
                                            loss_func = loss_func, verbose = False)
                accs.append(valid_result[0])
                losses_valid.append(valid_result[1])

                # learning rate descent strategy
                if len(accs) > 1 and accs[-1] < accs[-2]:
                    self.lr *= .5


        return {'loss': losses, 'acc': accs, 'loss_valid': losses_valid}
    
    # prediction 
    def predict(self, x, y = None, batch_size = 40, loss_func = None, verbose = True):
        if y is not None:
            batch_e = 0
            n = x.shape[0]
            acc = 0
            loss = 0
            iterator = range(n // batch_size)
            if verbose: iterator = tqdm(iterator)
            for _ in iterator:
                batch_s = batch_e 
                batch_e = batch_s + batch_size
                batch_x = x[batch_s: batch_e]
                batch_y = y[batch_s: batch_e]

                # forward
                predict_y = self.forward(batch_x)
                acc += np.sum(batch_y[np.arange(batch_size),np.argmax(predict_y, axis = -1)])
                loss += self.compute_loss(predict_y, batch_y, batch_size, loss_func)

            loss /= (n // batch_size) 

            if verbose: print(f'Acc = {int(acc)}/{n} = {acc * 100. / n}%')
            return acc * 1. / n , loss # return accuracy and loss

        else:
            batch_e = 0
            n = x.shape[0]
            acc , loss = 0 , 0
            predicts = np.zeros(1, dtype='int16')

            iterator = range(n // batch_size)
            if verbose: iterator = tqdm(iterator)
            for _ in iterator:
                batch_s = batch_e 
                batch_e = batch_s + batch_size
                batch_x = x[batch_s: batch_e]

                # forward
                predict_y = self.forward(batch_x)
                predict_y = np.argmax(predict_y, axis = -1).flatten()
                predicts = np.hstack((predicts, predict_y))
            return predicts[1:]

    @classmethod
    def load(self, path):
        def arrayload(x, f):
            n = x.shape[0]
            for i in range(n):
                x[i] = np.array([float(i) for i in f.readline()[:-1].split()])

        with open(path, 'r') as f:
            hidden_size = [int(i) for i in f.readline()[:-1].split()]
            acts = f.readline()[:-1].split()
            model = Classifier(hidden_size, acts)
            for i in range(len(hidden_size) - 1):
                arrayload(model.weights[i], f)
                arrayload(model.biases [i], f)
        return model

    # save model
    def save(self, path):
        def arraysave(x, f):
            for line in x:
                f.write('\n' + ' '.join(['%.6f'%value for value in line]))
        
        with open(path,'w') as f:
            f.write(' '.join([str(i) for i in self.hidden_size]))
            f.write('\n' + ' '.join([str(i) for i in self.acts]))
            for i in range(len(self.weights)):
                arraysave(self.weights[i], f)
                arraysave(self.biases [i], f)

if __name__ == '__main__':
    pass

