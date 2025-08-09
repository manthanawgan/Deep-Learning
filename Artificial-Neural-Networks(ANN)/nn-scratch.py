import numpy as np
import random 
import math

class Neuron:
    def __init__(self, nin):
        self.w = []
        for _ in range(nin):
            val = random.uniform(-1, 1)
            self.w.append(Value(val))
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out


class Layers:
    def __init__(self, nin, nout):
        self.neurons = []           #[Neuron(nin) for _ in range(nout)]
        for _ in range(nout):
            self.neurons.append(Neuron(nin))

    def __call__(self, x):
        outs = []
        for n in self.neurons:
            outs.append(n(x))
        return outs
    

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            nin = sz[i]
            nout = sz[i+1]
            #nonlin = i != len(nouts) - 1
            layer = Layers(nin, nout)  #layer = Layers(nin, nout, nonlin=nonlin)
            self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

