from Neuron import InputNeuron, Neuron, OutputNeuron

class Layer:
    pass

class InputLayer:
    def __init__(self, neurons: list[InputNeuron]):
        self.neurons = neurons

class Layer:
    def __init__(self, neurons: list[Neuron], prevLayer: Layer):
        self.neurons = neurons
        self.prevLayer = prevLayer
    
    def CalculateNeurons(self):
        # for i in range(len(self.neurons)):
        #     self.neurons[i].Calculate(self.prevLayer.neurons)
        
        for neuron in self.neurons:
            neuron.Calculate(self.prevLayer.neurons)

class OutputLayer:
    def __init__(self, neurons: list[OutputNeuron]):
        self.neurons = neurons
