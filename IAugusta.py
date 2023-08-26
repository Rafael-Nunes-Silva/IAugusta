from NeuralNetwork import NeuralNetwork
from PIL import Image
import numpy

def ImageToInputValues(imagePath: str) -> list[bool]:
    inputValues = []
    with Image.open(imagePath) as img:
        for color in numpy.array(img).reshape((400, 4)):
            inputValues.append(bool(int(color[0])))
    return inputValues



test = NeuralNetwork()
# test.StartNew(400, 10, 2, 16)
test.LoadLearnedData("test")

inputValues = ImageToInputValues("Images/0 - 1.png")
test.SetInputValues(inputValues)

print(test.RunInput())

# test.SaveLearnedData("test")
