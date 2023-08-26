class Neuron:
    pass

class InputNeuron:
    def __init__(self, active: bool):
        self.active = active

class Neuron:
    def __init__(self, weights: list[float], bias: float):
        self.active = False

        self.weights = weights
        self.bias = bias
    
    # Calculate whether the neuron will be activated or not based on the previous neuron layer
    def Calculate(self, prevNeurons: list[Neuron]):
        value = 0
        for i in range(len(prevNeurons)):
            value += int(prevNeurons[i].active) * self.weights[i]
        value += self.bias

        self.active = True if max(0, value) >= self.bias else False

class OutputNeuron:
    def __init__(self, weights: list[float], bias: float):
        self.weights = weights
        self.bias = bias

    # Calculate how much the network thinks the input is related to this neuron's "idea"
    def Calculate(self, prevNeurons: list[Neuron]):
        value = 0
        for i in range(len(prevNeurons)):
            value += int(prevNeurons[i].active) * self.weights[i]
        value += self.bias

        self.value = max(0, value)
        # self.active = True if max(0, value) >= self.bias else False
