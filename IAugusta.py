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
# test.StartNew(400, 10, 4, 32)               # Start the network with the desired parameters
# test.SaveLearnedData("testBeforeLearning")  # Save the random weights and biases for comparison after learning
test.LoadLearnedData("testBeforeLearning")

# Set the input values on the input neurons
inputValues = ImageToInputValues("Images/9 - 4.png")
test.SetInputValues(inputValues)

test.RunInput()         # Run input
print(test.GetOutput()) # Print output

# Train x times for the input given before
trainingExamples = [(ImageToInputValues(f"Images/9 - {random.randrange(1, 4)}.png"), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) for i in range(1000)]
# test.Train([(inputValues, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) for i in range(1000)])
test.Train(trainingExamples)

test.RunInput()         # Run input
print(test.GetOutput()) # Print output

# Save the weights and biases after learning for before and after
test.SaveLearnedData("testAfterLearning")
