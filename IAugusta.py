from NeuralNetwork import NeuralNetwork

test = NeuralNetwork()

test.StartNew(10, 10, 2, 4)
test.SaveLearnedData("test")

test.LoadLearnedData("test")

input()
