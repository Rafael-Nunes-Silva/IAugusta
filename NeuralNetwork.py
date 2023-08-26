from Layer import InputLayer, Layer, OutputLayer
from Neuron import InputNeuron, Neuron, OutputNeuron
import json
import random

class NeuralNetwork:
    layers = []

    def StartNew(self, inputsCount: int, outputsCount: int, networkDepth: int, layerWidth: int):
        self.inputsCount = inputsCount
        self.outputsCount = outputsCount
        self.networkDepth = networkDepth
        self.layerWidth = layerWidth

        self.__SetupInputLayer()
        self.__SetupLayers()
        self.__SetupOutputLayer()

    def SaveLearnedData(self, learnedDataPath: str):
        layers = []
        for i in range(1, len(self.layers) - 1):
            layers.append({
                "weights": [self.layers[i].neurons[j].weights for j in range(len(self.layers[i].neurons))],
                "biases": [self.layers[i].neurons[j].bias for j in range(len(self.layers[i].neurons))]
            })
        
        learnedData = {
            "inputsCount": self.inputsCount,
            "outputsCount": self.outputsCount,
            "networkDepth": self.networkDepth,
            "layerWidth": self.layerWidth,
            "layers": layers,
            "outputLayer": {
                "weights": [self.layers[-1].neurons[j].weights for j in range(len(self.layers[-1].neurons))],
                "biases": [self.layers[-1].neurons[j].bias for j in range(len(self.layers[-1].neurons))]
            }
        }

        with open(f"{learnedDataPath}.json", "w") as learnedDataFile:
            json.dump(learnedData, learnedDataFile)

    def LoadLearnedData(self, learnedDataPath: str):
        learnedDataJson = {}
        with open(f"{learnedDataPath}.json", "r") as learnedDataFile:
            learnedDataJson = json.loads(learnedDataFile.read())

        self.inputsCount = learnedDataJson["inputsCount"]
        self.outputsCount = learnedDataJson["outputsCount"]
        self.networkDepth = learnedDataJson["networkDepth"]
        self.layerWidth = learnedDataJson["layerWidth"]

        # Creating the layers based on the saved learning data
        self.__SetupInputLayer()
        self.__SetupLayersFromSavedValues(learnedDataJson)
        self.__SetupOutputLayerFromSavedValues(learnedDataJson)

    def SetInputValues(self, activations: list[bool]):
        neurons = []

        for i in range(self.inputsCount):
            neurons.append(InputNeuron(activations[i]))

        return InputLayer(neurons)

    def __SetupInputLayer(self):
        self.layers = [InputLayer([InputNeuron(False) for i in range(self.inputsCount)])]

    def __SetupLayers(self):
        for i in range(self.networkDepth):
            neurons = []
            for j in range(self.layerWidth):
                randomWeights = [random.uniform(-1.0, 1.0) for k in range(self.layerWidth)]
                randomBiases = [random.uniform(0.0, 1.0) for k in range(self.layerWidth)]
                neurons.append(Neuron(randomWeights, randomBiases))
            self.layers.append(Layer(neurons, self.layers[-1]))

    def __SetupOutputLayer(self):
        neurons = []
        for i in range(self.outputsCount):
            randomWeights = [random.uniform(-1.0, 1.0) for k in range(self.layerWidth)]
            randomBiases = [random.uniform(0.0, 1.0) for k in range(self.layerWidth)]
            neurons.append(OutputNeuron(randomWeights, randomBiases))
        self.layers.append(OutputLayer(neurons))

    def __SetupLayersFromSavedValues(self, learnedDataJson: dict):
        for i in range(self.networkDepth):
            neurons = []
            for j in range(self.layerWidth):
                bias = learnedDataJson["layers"][i]["biases"][j]
                neurons.append(Neuron(learnedDataJson["layers"][i]["weights"], bias))
            # self.layers.append(Layer(neurons, (self.inputLayer if i == 0 else self.layers[-1])))
            self.layers.append(Layer(neurons, (self.layers[0] if i == 0 else self.layers[-1])))

    def __SetupOutputLayerFromSavedValues(self, learnedDataJson: dict):
        neurons = []
        for i in range(self.outputsCount):
            neurons.append(Neuron(learnedDataJson["outputLayer"]["weights"][i], learnedDataJson["outputLayer"]["biases"][i]))
        self.layers.append(Layer(neurons, self.layers[-1]))
