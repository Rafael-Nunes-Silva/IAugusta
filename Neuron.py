class Neuron:
    pass

# Input neuron, can only be active or inactive and doesn't calculate anything
class InputNeuron:
    value = 1
    def __init__(self, active: bool):
        self.active = int(active)

# Main neuron, responsible for the calculations and can only be active or inactive
class Neuron:
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias
    
    # Calculate the activation of the neuron based on the previous neuron layer
    def Calculate(self, prevNeurons: list[Neuron]):
        value = 0
        for i in range(len(prevNeurons)):
            value += prevNeurons[i].value * self.weights[i]
        value += self.bias

        self.value = min(max(0, value), 1)

    def AjustWeights(self, weightChanges):
        for i in range(len(self.weights)):
            self.weights[i] += weightChanges[i]

# Output neuron, also does calculations but can be inactive (0), or any value number
class OutputNeuron(Neuron):
    def __init__(self, weights: list[float], bias: float):
        Neuron.__init__(self, weights, bias)
