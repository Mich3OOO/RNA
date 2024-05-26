import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
class capa:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class activacion:
    def __init__(self, activation, activation_prime):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class red:
    def __init__(self):
        self.capas = []
        self.f = None
        self.ff = None
    
    
    def add(self,layer):
        self.capas.append(layer)

    def setf(self,f,ff):
        self.f = f
        self.ff = ff

    def predict(self,x):
        result = []

        for i in range(len(x)):
            salida = x[i]
            for cp in self.capas:
                salida = cp.forward_propagation(salida)
            result.append(salida)
        return result

    

    def fit(self,x,y,times,learning_rate):
        

        cm = plt.get_cmap('gist_rainbow')
        for i in range(times):

            err = 0

            for j in range(len(x)):
                salida = x[j]
                for cp in self.capas:
                    salida = cp.forward_propagation(salida)
                err += self.f(y[j],salida)
                plt.plot(i,self.f(y[j],salida),color=cm(50*j),marker='.')
                error = self.ff(y[j],salida)
                for cp in reversed(self.capas):
                    error = cp.backward_propagation(error,learning_rate)

                
            
            err /= len(x)
            
            
            print("training ... %f   error=%f" % (((i+1)/(times))*100, err))
        plt.show()

def sigmoide(x):
    return 1/(1+np.exp(-x))

def derivadasigmoide(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

def error(y, yy):
    return np.mean(np.power(y-yy, 2))

def error_derivada(y, yy):
    return 2*(yy-y)/y.size

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size













#or

n = red()
n.setf(mse, mse_prime)
n.add(capa(2,3))
n.add(activacion(sigmoide, derivadasigmoide))
n.add(capa(3,3))
n.add(activacion(sigmoide, derivadasigmoide))
n.add(capa(3,1))
n.add(activacion(sigmoide, derivadasigmoide))

a = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
aa =np.array([[[0]], [[1]],[[1]],[[0]]])
b = np.array([[[1,0]],[[0,0]],[[0,1]]])
bb = np.array([[[1]],[[0]],[[1]]])
n.fit(b, bb, 2000, 0.2)
for t in a:

    print(t,n.predict(t))
