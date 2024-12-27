import scipy.stats
import torch.nn as nn

'''
Dieses Skript definiert die torch-Modelle, die zum Training gebraucht werden.
'''

act = nn.LeakyReLU
act2 = nn.ReLU
act3 = nn.Sigmoid

# Wichtig: Die Erste Zahl den self.linear_layer_1 muss zur Inputdimension der
# Daten passen und die letzte Zahl des letzten layers muss zur Zahl der Klassen
# passen. Die Layer müssen immer mit einer Aktivierungsfunktion versehen werden
# und die Listen self.linear_layers und self.activation_layers müssen befüllt
# werden. 

# Am Einfachsten ist es keine neue eigene class Model_X Klasse zu schreiben,
# da hierfür dann das Training_NN Skript geändert werden muss, sondern lediglich eine
# bestehende Klasse auf eure Bedürfnisse anzupassen.

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.linear_layers = []
        self.activation_layers = []
        self.linear_layer_1 = nn.Linear(6,10)
        self.linear_layer_2 = nn.Linear(10,4)
        #self.linear_layer_3 = nn.Linear(10,4)
        self.linear_layers.append(self.linear_layer_1)
        self.linear_layers.append(self.linear_layer_2)
        #self.linear_layers.append(self.linear_layer_3)
        self.activation_layers_1 = act2()
        #self.activation_layers_2 = act2()
        self.activation_layers_2 = nn.Softmax(-1)
        self.activation_layers.append(self.activation_layers_1)
        self.activation_layers.append(self.activation_layers_2)
        #self.activation_layers.append(self.activation_layers_3)
        self.num_layers = len(self.linear_layers)
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.linear_layers[i](x)
            x = self.activation_layers[i](x)
        return x

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.linear_layers = []
        self.activation_layers = []
        self.linear_layer_1 = nn.Linear(5,8)
        self.linear_layer_2 = nn.Linear(8,8)
        self.linear_layer_3 = nn.Linear(8,2)
        self.linear_layers.append(self.linear_layer_1)
        self.linear_layers.append(self.linear_layer_2)
        self.linear_layers.append(self.linear_layer_3)
        self.activation_layers_1 = act2()
        self.activation_layers_2 = act2()
        self.activation_layers_3 = nn.Softmax(-1)
        self.activation_layers.append(self.activation_layers_1)
        self.activation_layers.append(self.activation_layers_2)
        self.activation_layers.append(self.activation_layers_3)
        self.num_layers = len(self.linear_layers)
    def forward(self, y):
        for i in range(self.num_layers):
            x = self.linear_layers[i](y)
            y = self.activation_layers[i](x)
        return x
    
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.linear_layers = []
        self.activation_layers = []
        self.linear_layer_1 = nn.Linear(5,8)
        self.linear_layer_2 = nn.Linear(8,8)
        self.linear_layer_3 = nn.Linear(8,3)
        self.linear_layers.append(self.linear_layer_1)
        self.linear_layers.append(self.linear_layer_2)
        self.linear_layers.append(self.linear_layer_3)
        self.activation_layers_1 = act2()
        self.activation_layers_2 = act2()
        self.activation_layers_3 = nn.Softmax(-1)
        self.activation_layers.append(self.activation_layers_1)
        self.activation_layers.append(self.activation_layers_2)
        self.activation_layers.append(self.activation_layers_3)
        self.num_layers = len(self.linear_layers)
    def forward(self, y):
        for i in range(self.num_layers):
            x = self.linear_layers[i](y)
            y = self.activation_layers[i](x)
        return x

class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        self.linear_layers = []
        self.activation_layers = []
        self.linear_layer_1 = nn.Linear(4,8)
        self.linear_layer_2 = nn.Linear(8,8)
        self.linear_layer_3 = nn.Linear(8,3)
        self.linear_layers.append(self.linear_layer_1)
        self.linear_layers.append(self.linear_layer_2)
        self.linear_layers.append(self.linear_layer_3)
        self.activation_layers_1 = act2()
        self.activation_layers_2 = act2()
        self.activation_layers_3 = nn.Softmax(-1)
        self.activation_layers.append(self.activation_layers_1)
        self.activation_layers.append(self.activation_layers_2)
        self.activation_layers.append(self.activation_layers_3)
        self.num_layers = len(self.linear_layers)
    def forward(self, y):
        for i in range(self.num_layers):
            x = self.linear_layers[i](y)
            y = self.activation_layers[i](x)
        return x
