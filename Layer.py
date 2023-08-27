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
        for i in range(len(self.neurons)):
            self.neurons[i].Calculate(self.prevLayer.neurons)
    
    def CalculateNeuronDesires(self, neuronsDesires) -> list[float]:
        weightChanges = [0 for x in range(len(self.prevLayer.neurons))]

        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i].weights)):
                weightChanges[j] += neuronsDesires[i] * self.neurons[i].weights[j] * self.prevLayer.neurons[j].value
                self.neurons[i].weights[j] += weightChanges[j] / len(self.neurons)

        for k in range(len(weightChanges)):
            weightChanges[k] /= len(self.neurons)
        
        # for i in range(len(self.neurons)):
        #     self.neurons[i].AjustWeights(weightChanges)
        
        return weightChanges

# Output layer, holds the output neurons and a reference to the previous layer
class OutputLayer(Layer):
    def __init__(self, neurons: list[OutputNeuron], prevLayer: Layer):
        Layer.__init__(self, neurons, prevLayer)
    
    def CalculateNeuronDesires(self, expectedOutput: list[float]) -> list[float]:
        weightChanges = [0 for x in range(len(self.prevLayer.neurons))]

        for i in range(len(self.neurons)):
            neuronDesire = expectedOutput[i] - self.neurons[i].value

            for j in range(len(self.neurons[i].weights)):
                weightChanges[j] += neuronDesire * self.neurons[i].weights[j] * self.prevLayer.neurons[j].value
                self.neurons[i].weights[j] += weightChanges[j]

        for k in range(len(weightChanges)):
            weightChanges[k] /= len(self.neurons)
        
        # for i in range(len(self.neurons)):
        #     self.neurons[i].AjustWeights(weightChanges)
        
        return weightChanges
