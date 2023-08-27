from NeuralNetwork import NeuralNetwork
from PIL import Image
import numpy
import random

def ImageToInputValues(imagePath: str) -> list[bool]:
    inputValues = []
    with Image.open(imagePath) as img:
        for color in numpy.array(img).reshape((400, 4)):
            inputValues.append(not bool(int(color[0])))
    return inputValues



test = NeuralNetwork()                      # Create an instance of the network
# test.StartNew(400, 10, 2, 16)               # Start the network with the desired parameters
# test.SaveLearnedData("testBeforeLearning")  # Save the random weights and biases for comparison after learning
test.LoadLearnedData("testBeforeLearning")
# test.LoadLearnedData("testAfterLearning")

trainingExamples = [
    # ("Images/0 - 1.png", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/0 - 2.png", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/0 - 3.png", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/1 - 1.png", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/1 - 2.png", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/1 - 3.png", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/2 - 1.png", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/2 - 2.png", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/2 - 3.png", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/3 - 1.png", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    # ("Images/3 - 2.png", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    # ("Images/3 - 3.png", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    # ("Images/4 - 1.png", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    # ("Images/4 - 2.png", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    # ("Images/4 - 3.png", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    # ("Images/5 - 1.png", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    # ("Images/5 - 2.png", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    # ("Images/5 - 3.png", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    # ("Images/6 - 1.png", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    # ("Images/6 - 2.png", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    # ("Images/6 - 3.png", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    # ("Images/7 - 1.png", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    # ("Images/7 - 2.png", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    # ("Images/7 - 3.png", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    # ("Images/8 - 1.png", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    # ("Images/8 - 2.png", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    # ("Images/8 - 3.png", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    ("Images/9 - 1.png", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    ("Images/9 - 1.png", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    ("Images/9 - 1.png", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
]

# trainingData = [(ImageToInputValues(trainingExamples[i % 10][0]), trainingExamples[1 + i % 3][1]) for i in range(10000)]
# trainingData = []
# for i in range(1000):
#     exampleIndex = i % len(trainingExamples)
#     digit = trainingExamples[exampleIndex][0]
#     output = trainingExamples[exampleIndex][1]
#     trainingData.append((ImageToInputValues(digit), output))

test.Train([(ImageToInputValues(trainingExamples[i][0]), trainingExamples[i][1]) for i in range(len(trainingExamples))])
# test.SaveLearnedData("testAfterLearning")

testingExamples = [
    # ("Images/0 - 4.png", [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/1 - 4.png", [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/2 - 4.png", [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    # ("Images/3 - 4.png", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    # ("Images/4 - 4.png", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    # ("Images/5 - 4.png", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    # ("Images/6 - 4.png", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    # ("Images/7 - 4.png", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    # ("Images/8 - 4.png", [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    ("Images/9 - 1.png", [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
]

for testExample in testingExamples:
    print(f"\ninput: {testExample[0]}")
    test.SetInputValues(ImageToInputValues(testExample[0]))
    test.RunInput() # Run input
    print(f"output: {test.GetOutput()}")    # Print output
    print(f"desired output: {testExample[1]}")
