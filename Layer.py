from Neuron import InputNeuron, Neuron, OutputNeuron

class Layer:
    pass

# Input layer responsible for holding the input neurons
class InputLayer:
    def __init__(self, neurons: list[InputNeuron]):
        self.width = len(neurons)
        
        self.neurons = neurons

# Main layer, holds it's neurons and a refetence to the previous layer
class Layer:
    def __init__(self, neurons: list[Neuron], prevLayer: Layer):
        self.width = len(neurons)

        self.neurons = neurons
        self.prevLayer = prevLayer
    
    def CalculateNeurons(self):
        # for i in range(len(self.neurons)):
        #     self.neurons[i].Calculate(self.prevLayer.neurons)
        
        for neuron in self.neurons:
            neuron.Calculate(self.prevLayer.neurons)

# Output layer, holds the output neurons and a reference to the previous layer
class OutputLayer:
    def __init__(self, neurons: list[OutputNeuron], prevLayer: Layer):
        self.width = len(neurons)
        
        self.neurons = neurons
        self.prevLayer = prevLayer
    
    def CalculateNeurons(self):
        # for i in range(len(self.neurons)):
        #     self.neurons[i].Calculate(self.prevLayer.neurons)
        
        for neuron in self.neurons:
            neuron.Calculate(self.prevLayer.neurons)
