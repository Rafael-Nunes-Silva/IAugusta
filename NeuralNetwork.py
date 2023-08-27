from Layer import InputLayer, Layer, OutputLayer
from Neuron import InputNeuron, Neuron, OutputNeuron
import json
import random

class NeuralNetwork:
    layers = []

    # Sets up the network with new user set values and random weights and biases
    def StartNew(self, inputsCount: int, outputsCount: int, networkDepth: int, layerWidth: int):
        self.inputsCount = inputsCount
        self.outputsCount = outputsCount
        self.networkDepth = networkDepth
        self.layerWidth = layerWidth

        self.__SetupInputLayer()
        self.__SetupLayers()
        self.__SetupOutputLayer()

    # Saves the data used by the network, input and output sizes, depth, layerWidth and all the weights and biases into a json file
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

    # Load's the network data, input and output sizes, depth, layerWidth and all the weights and biases, from a json file
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

    # Runs the network and returns the list of values from the output neurons
    def RunInput(self):
        for i in range(1, len(self.layers)):
            self.layers[i].CalculateNeurons()

    def GetOutput(self) -> list[float]:
        return [self.layers[-1].neurons[i].value for i in range(self.outputsCount)]

    # Trains the network with the specified list of inputs
    def Train(self, input_output: list[tuple[list[bool], list[float]]]):
        learnedValues = [[] for i in range(self.networkDepth + 1)]
        for i in range(len(learnedValues)):
            learnedValues[i] = [0 for j in range(len(self.layers[i].neurons))]

        # progress = 0
        for input, expectedOutput in input_output:
            desiredChanges = self.__TrainInput(input, expectedOutput)[::-1]
            for i in range(len(desiredChanges)):
                for j in range(len(desiredChanges[i])):
                    learnedValues[i][j] += desiredChanges[i][j]
        
            # progress += 1
            # print(f"{progress}/{len(input_output)}")
        
        # for i in range(len(learnedValues)):
        #     for j in range(len(learnedValues[i])):
        #         for k in range(len(self.layers[i + 1].neurons[j].weights)):
        #             self.layers[i + 1].neurons[j].weights[k] *= learnedValues[i][j]

    # Set's the values for the input neurons
    def SetInputValues(self, activations: list[bool]):
        neurons = []

        for i in range(self.inputsCount):
            neurons.append(InputNeuron(activations[i]))

        return InputLayer(neurons)

    # Creates an input layer with all the neurons inactive
    def __SetupInputLayer(self):
        self.layers = [InputLayer([InputNeuron(False) for i in range(self.inputsCount)])]

    # Creates the layers between the input and the output with random values for the neuron's weights and biases
    def __SetupLayers(self):
        for i in range(self.networkDepth):
            neurons = []
            for j in range(self.layerWidth):
                randomWeights = [random.uniform(-1.0, 1.0) for k in range(self.layers[i].width)]
                neurons.append(Neuron(randomWeights, random.uniform(0.0, 1.0)))
            self.layers.append(Layer(neurons, self.layers[-1]))

    # Creates the output layer with random values for the neuron's weights and biases
    def __SetupOutputLayer(self):
        neurons = []
        for i in range(self.outputsCount):
            randomWeights = [random.uniform(-1.0, 1.0) for k in range(self.layers[-1].width)]
            neurons.append(OutputNeuron(randomWeights, random.uniform(0.0, 1.0)))
        self.layers.append(OutputLayer(neurons, self.layers[-1]))

    # Creates the layers between the input and the output with the loaded values for the neuron's weights and biases
    def __SetupLayersFromSavedValues(self, learnedDataJson: dict):
        for i in range(self.networkDepth):
            neurons = []
            for j in range(self.layerWidth):
                bias = learnedDataJson["layers"][i]["biases"][j]
                neurons.append(Neuron(learnedDataJson["layers"][i]["weights"][j], bias))
            # self.layers.append(Layer(neurons, (self.inputLayer if i == 0 else self.layers[-1])))
            self.layers.append(Layer(neurons, (self.layers[0] if i == 0 else self.layers[-1])))

    # Creates the output layer with the loaded values for the neuron's weights and biases
    def __SetupOutputLayerFromSavedValues(self, learnedDataJson: dict):
        neurons = []
        for i in range(self.outputsCount):
            neurons.append(OutputNeuron(learnedDataJson["outputLayer"]["weights"][i], learnedDataJson["outputLayer"]["biases"][i]))
        self.layers.append(OutputLayer(neurons, self.layers[-1]))

    def __CalculateCost(self, desiredOutput: list[float]):
        cost = 0
        for i in range(self.outputsCount):
            cost += pow((self.layers[-1].neurons[i].value - desiredOutput[i]), 2)

        return cost

    def __TrainInput(self, input: list[bool], expectedOutput: list[float]) -> list[list[float]]:
        self.SetInputValues(input)
        self.RunInput()

        neuronsDesires = [self.layers[-1].CalculateNeuronDesires(expectedOutput)]
        for i in range(2, len(self.layers)):
            neuronsDesires.append(self.layers[-i].CalculateNeuronDesires(neuronsDesires[-1]))
        
        return neuronsDesires
        # print("changes to last layer's weights:", self.layers[-1].CalculateNeuronDesires(expectedOutput))

    def __CalculateLayerDesires() -> list[list[float]]:
        pass
