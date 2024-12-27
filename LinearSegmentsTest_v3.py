# -*- coding: utf-8 -*-
import numpy
import torch
import numpy as np
import copy
from itertools import chain, combinations



#simplified version of the model class found in PropagationBase.py
#should be easy to use the already implemented Model class instead
#alpha is a parameter: alpha=0.0 gives us the ReLu case 
class Model():
    
    def __init__(self, weights, biases, alpha=0.0):
        self.weights = weights
        self.biases = biases
        self.num_layers = len(weights)
        self.alpha = alpha 
    
    def propagate_with_start_layer(self, point, start_layer):
        for i in range(start_layer, self.num_layers):
            point = np.matmul(self.weights[i], point) + self.biases[i]
           
            #if i in range(0, self.num_layers-1):
            if i < self.num_layers - 1:
                for j in range(0, self.weights[i].shape[0]):
                    if point[j]<0:
                        point[j] = self.alpha * point[j]
        return point


                        
    #propagates a list of points through the model; each hidden layer is followed by an 
    #activation layer by a leaky ReLu with self.alpha as parameter 
    #need to write a function for PropagationBase.py which uses activation function already implemented 
    #points must be numpy arrays of size >=2
    def propagate(self, points):
        output_points = []
        for point in points:
            for i in range(0, self.num_layers):
                point = np.matmul(self.weights[i], point) + self.biases[i]
               
                if i < self.num_layers - 1:
                    for j in range(0, self.weights[i].shape[0]):
                        if point[j]<0:
                            point[j] = self.alpha * point[j]
        
            output_points.append(point)
        
        return output_points

    def predict(self, points):
        points = points.values.tolist()
        return self.propagate(points)

    def predict_proba(self, points):
        if type(points) is not numpy.ndarray:
            points = points.values.tolist()
        predictions = self.propagate(points)
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        return probabilities




#generate n-dim. Cube with connections
#generiert Einheitswürfel, falls nicht gestreckt oder gestaucht wird 
#Ecken haben Indizes 0 oder 1 in allen Dimensionen, falls keine Translationen 
#oder Streckungen vorliegen 
#nutze diese Klasse, um Würfel als CPU-Objekt zu erstellen 
class Cube():
    
    def __init__(self, dimension):
        self.dimension = dimension
        
        points = []
        ps = powerset(range(self.dimension))
        for subset in ps:
            point = np.zeros(self.dimension)
            for l in subset:
                point[l] = 1
            points.append(point)
        
        self.points = points
        
        conns = []
        for k, pt1 in enumerate(self.points):
            for l, pt2 in enumerate(self.points[k+1:]):
                if np.sum(abs(pt1-pt2)) == 1:
                    conns.append((k, k+l+1))
        
        self.conn = set(conns)
    
    #Translation und Streckung/Stauchung der Punkte in einer Dimension
    def translate(self, b, dim):
        for pt in self.points:
            pt[dim-1] += b
            
        
    def stretch(self, a, dim):
        for pt in self.points:
            pt[dim-1] *= a


#Hilfsfunktion für Potenzmenge 
def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    
#load model saved in a .pt file 
def loadModel(file, alpha=0.01):
    
    param_dict = torch.load(file) #"NN1.pt"
    w = []
    b= []
    num_layers = len({int(item.split("_")[1]) for item in param_dict})
    for i in range(num_layers):
            w.append(param_dict["weights_" + str(i)].numpy())
            b.append(param_dict["bias_" + str(i)].numpy())

    model = Model(w, b, alpha=alpha)
    return model 

def loadModel_AS(file, alpha=0):
    
    param_dict = file
    w = []
    b= []
    try:
        num_layers = len({int(item.split("_")[1]) for item in param_dict})
    except:
        num_layers = len({int(re.findall(r'\d+', item)[0]) for item in param_dict})
    #print(num_layers)
    for i in range(num_layers):
        try:
            w.append(param_dict["weights_" + str(i)].numpy())
            b.append(param_dict["bias_" + str(i)].numpy())
        except:
            w.append(param_dict["fc" + str(i+1) + ".weight"].numpy())
            b.append(param_dict["fc" + str(i+1) + ".bias"].numpy())
    model = Model(w, b, alpha=alpha)
    return model 
    
#find linear segments of the model for the given cpu as input polytope 
def getLinearSegments(model, cpu, alpha = 0.0):   
    
    #cpu is an element of class CPU
    #call its linear activation and divide by activation functions in right order
    #return the cpu element 
    for i in range(0, model.num_layers):
        #print(f"Layer number {i+1} of {model.num_layers}")
        cpu.linear_transform(model.weights[i], model.biases[i])
        
        dimension = model.weights[i].shape[0]
        if i in range(0, model.num_layers-1):
            for j in range(0, dimension):
                #print(f"Neuron number {j+1} of {dimension}")
                cpu.divide_by_activation(j+1, alpha)
        
    return cpu 

def get_restricted_cpu(model, cpu, cert_vec = None, output_vec = None, alpha = 0.0):
    adjusted_weights = copy.deepcopy(model.weights)
    adjusted_biases = copy.deepcopy(model.biases)
        
    if cert_vec is not None:
        uncer_idx = [i for i,v in enumerate(cert_vec) if v == None]
        
        if len(uncer_idx) != len(cpu.input_points_union[0]):
            print(uncer_idx,len(cpu.input_points_union[0]))
            raise ValueError("ill-matched certainty dimensions")
        
        w1_shape = model.weights[0].shape
        new_weight = np.zeros((w1_shape[0], len(uncer_idx)))
        
        for i in range(w1_shape[0]):
            none_count = 0
            for j, v in enumerate(cert_vec):
                if v is None:
                    new_weight[i][none_count] = model.weights[0][i][j]
                    none_count +=1
                else:
                    adjusted_biases[0][i] += model.weights[0][i][j]*v
                    
        adjusted_weights[0] = new_weight
    
    if output_vec is not None:
        out_cer_idx = [i for i,v in enumerate(output_vec) if v != None]
        
        new_out_weight = np.zeros((len(out_cer_idx), model.weights[-1].shape[1]))
        new_out_bias = np.zeros(len(out_cer_idx),)
        new_out_weight = model.weights[-1][out_cer_idx]
        new_out_bias = model.biases[-1][out_cer_idx]
        
        adjusted_weights[-1] = new_out_weight
        adjusted_biases[-1] = new_out_bias
        
    for i in range(0, model.num_layers):
        #print(f"Layer number {i+1} of {model.num_layers}")
        cpu.linear_transform(adjusted_weights[i], adjusted_biases[i])
        
        if i in range(0, model.num_layers-1):
            dimension = adjusted_weights[i].shape[0]
            for j in range(0, dimension):
                #print(f"Neuron number {j+1} of {dimension}")
                cpu.divide_by_activation(j+1, alpha)

    return cpu 



